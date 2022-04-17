#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "fftw/fftw3.h"

int i32min(int a, int b){return (a>b)?b:a;}
int i32max(int a, int b){return (a>b)?a:b;}
int int24toint32(int32_t* ptr)
{
    //endianness-dependent
    int res = 0;
    res |= (*(ptr+0) & 0xFF) << 0;
    res |= (*(ptr+1) & 0xFF) << 8;
    res |= (*(ptr+2) & 0xFF) << 16;
    if ((res & 0x00800000) > 0)
        res |= 0xFF000000;
    else
        res &= 0x00FFFFFF;
    return res;
}
struct lppeak
{
    lppeak(float v, int va, int vb)
    {
        value=v;
        a=va;
        b=vb;
    }
    lppeak()
    {
        lppeak(0,0,0);
    }
    int a,b;
    float value;
};

float blackman_harris(float x)
{
    float res = 0.35875;
    res -= 0.48829 * cosf(2*3.14159265358979*x);
    res += 0.14128 * cosf(4*3.14159265358979*x);
    res -= 0.01168 * cosf(6*3.14159265358979*x);
    return res;
}

int main(int argc, char** argv)
{
    int FFTSize = 2048;
    float* in_r2c = (float*)fftwf_malloc(2*FFTSize*sizeof(float));
    fftwf_complex* out_r2c = (fftwf_complex*)fftwf_malloc(4*(FFTSize/2+1)*sizeof(fftwf_complex));
    fftwf_plan p_r2c = fftwf_plan_dft_r2c_1d(FFTSize, in_r2c, out_r2c, FFTW_MEASURE);

    float* out_c2r = (float*)fftwf_malloc(2*FFTSize*sizeof(float));
    fftwf_complex* in_c2r = (fftwf_complex*)fftwf_malloc(4*(FFTSize/2+1)*sizeof(fftwf_complex));
    fftwf_plan p_c2r = fftwf_plan_dft_c2r_1d(FFTSize, in_c2r, out_c2r, FFTW_MEASURE);

    char fn_in[512];
    char fn_in_lp[512];
    char fn_out[512];
    float gain = 6;
    float postgain = 0;
    int boxblurpasses = 3;
    int boxblurradius = 64;
    float cutofffreq = 190.0;

    printf("original file: ");
    scanf("%s", fn_in);
    printf("output file: ");
    scanf("%s", fn_out);
    printf("gain (dB): ");
    scanf("%f", &gain);
    printf("post-gain (dB): ");
    scanf("%f", &postgain);
    printf("smoothing radius (samples): ");
    scanf("%d", &boxblurradius);
    printf("filter cutoff freq (Hz): ");
    scanf("%f", &cutofffreq);

    //dB -> coeff
    gain = pow(10.0, gain/20.0);
    postgain = pow(10.0, postgain/20.0);

    int16_t* lp_peaks = (int16_t*)calloc(44100, sizeof(int16_t));

    int channels = 2;
    int smprate = 44100;
    int bitspersmp = 16;
    int adSize = 0;

    std::ifstream af_in(fn_in, std::ios::binary);
    std::ofstream af_out(fn_out, std::ios::binary);

    if(!af_in.good() || !af_out.good())
        return 311;

    char hdr_in[44];
    af_in.read(hdr_in, 44);

    channels =      *(int16_t*)(&hdr_in[22]);
    smprate =       *(int32_t*)(&hdr_in[24]);
    bitspersmp =    *(int16_t*)(&hdr_in[34]);
    adSize =        *(int32_t*)(&hdr_in[40]);
    int smpcount = adSize/(channels*(bitspersmp/8));
    std::vector<lppeak> peaks[channels];

    printf("%d - %d-bit - %s\n", smprate, bitspersmp, (channels==1)?"Mono":"Stereo");
    printf("length: %d samples (%02d:%02d:%02d)\n", smpcount, smpcount/(smprate*3600), smpcount/(smprate*60), (smpcount/smprate)%60);
    printf("reading audio data... \n");

    int adAllocSize = smpcount*channels*sizeof(float);
    while(adAllocSize%(FFTSize*channels*sizeof(float)))
        adAllocSize++;
    void* aDataRaw = malloc(adSize);
    void* aDataOutRaw;
    float* aData;
    float* aDataLP;
    float* aDataOut;
    float* smpGain;
    af_in.read((char*)aDataRaw, adSize);

	aData = (float*)malloc(adAllocSize);

    if(bitspersmp == 8)
    {
        //assuming 8-bit unsigned integer
        uint8_t* cp = (uint8_t*)aDataRaw;
        for(int ch=0; ch<channels; ch++)
            for(int i=0; i<smpcount; i++)
                aData[i*channels+ch] = (float)(cp[i*channels+ch]) / 127.5 - 1.0;
    }
    if(bitspersmp == 16)
    {
        //assuming 16-bit signed integer
        int16_t* cp = (int16_t*)aDataRaw;
        for(int ch=0; ch<channels; ch++)
            for(int i=0; i<smpcount; i++)
                aData[i*channels+ch] = (float)(cp[i*channels+ch]) / 32768.0;
    }
    if(bitspersmp == 24)
    {
        //assuming 24-bit signed integer
        for(int ch=0; ch<channels; ch++)
            for(int i=0; i<smpcount; i++)
                aData[i*channels+ch] = (float)int24toint32((int32_t*)(aDataRaw+3*(i*channels+ch))) / 8388608.0;
    }
    if(bitspersmp == 32)
    {
        //assuming 32-bit float
        memcpy(aData, aDataRaw, adSize);
    }

    free(aDataRaw);
    aDataLP = (float*)malloc(adAllocSize);

    printf("FFT filtering... ");

    for(int ch=0; ch<channels; ch++)
    {
    	/*
    	phase 0:
			performs an FFT on 2048-sample blocks, zeroes out the bins that
			represent frequencies >cutofffreq and then performs an inverse
			FFT on that block, which is later put into aDataLP. the expected
			result from the iFFT is a low-pass filtered signal.

			it is generally not considered a good idea to filter signals by
			directly manipulating them in the frequency domain. however,
			this approach preserves the original phase of the waves and
			does not introduce delay, unlike most digital filters. this is
			important because of the use of the filtered signal's zero-crossings
			to time volume manipulation in this program.
    	*/

        printf("ch%dph0 ", ch);
        for(int i=0; i<smpcount; i+=FFTSize)
        {
            for(int j=0; j<FFTSize; j++)
                in_r2c[j] = aData[(i+j)*channels+ch];
            fftwf_execute(p_r2c);
            for(int j=0; j<FFTSize/2+1; j++)
                if((float)j*((float)smprate/(float)FFTSize) > cutofffreq || j == 0)
					out_r2c[j][0] = out_r2c[j][1] = 0.0;
            memcpy(in_c2r, out_r2c, 2*(FFTSize/2+1)*sizeof(fftwf_complex));
            fftwf_execute(p_c2r);
            for(int j=0; j<FFTSize; j++)
				aDataLP[(i+j)*channels+ch] = out_c2r[j];
        }
        printf("ch%dph1, ", ch);

        /*
        phase 1:
			due to FFT's assumptions about repetition of the input, signal
			in aDataLP lacks continuity. phase 1 mostly solves this problem
			by essentially repeating phase 0, but instead working on blocks
			shifted to the right by FFTSize/2 so that their middle is exactly
			at the intersection of phase 0 blocks. the signal is processed
			in exactly the same way as in phase 0, and then blended with
			the original output using a weighted average, where the weight
			is a window function.

			the triangle window is currently in use in this program for
			performance reasons, and because it is good enough for the limiter.
			when using this part of the program for EQ, a switch to the
			blackman-harris window should be made. (commented out below)
        */
        for(int i=FFTSize/2; i<smpcount-FFTSize; i+=FFTSize)
        {
            for(int j=0; j<FFTSize; j++)
                in_r2c[j] = (float)aData[(i+j)*channels+ch];
            fftwf_execute(p_r2c);
            for(int j=0; j<FFTSize/2+1; j++)
                if((float)j*((float)smprate/(float)FFTSize) > cutofffreq || j == 0)
					out_r2c[j][0] = out_r2c[j][1] = 0.0;
            memcpy(in_c2r, out_r2c, 2*(FFTSize/2+1)*sizeof(fftwf_complex));
            fftwf_execute(p_c2r);
            for(int j=0; j<FFTSize; j++)
            {
            	/* 0. triangle windowing */
                float weight = ((float)abs(j-FFTSize/2)/(float)(FFTSize/2));

                /* 1. blackman-harris windowing */
                //float weight = 1.0-blackman_harris((float)j/(float)FFTSize);

                aDataLP[(i+j)*channels+ch] = weight*aDataLP[(i+j)*channels+ch] + (1.0-weight)*out_c2r[j];
            }
        }


		/* normalizing the output: since FFTW computes an unnormalized DFT
		  (missing 1/sqrt(n) coeff in front of the summation) computing the
		  FFT and then the iFFT increases the output amplitude n-fold.
		  this step compensates for that, but is not needed as long as
		  the LP signal is only used for limiter timing purposes.
		*/
        /*
        for(int i=0; i<smpcount; i++)
            aDataLP[i*channels+ch] /= (float)FFTSize;
		*/
    }
    /*
    af_out.write(hdr_in, 44);
    af_out.write((char*)aDataLP, adSize);
    return 0;*/


    printf("\nfinding peaks...\n");

    for(int ch=0; ch<channels; ch++)
    {
        for(int i=0; i<smpcount; i++)
        {
            float sv = aDataLP[i*channels+ch];
            float peak = 1.0 / 262144.0; //to avoid division by zero
            int cs_beg = i;
            int cs_end = i;
            bool s = (sv>=0);
            for(int j=cs_beg; bool(aDataLP[j*channels+ch]>=0)==s && j<smpcount && j-cs_beg<=4096; j++)
            {
                cs_end = j;
                peak = std::max(peak, std::abs(aData[j*channels+ch]));
            }
            i = cs_end;
            peaks[ch].push_back(lppeak(peak, cs_beg, cs_end));
        }
    }
    free(aDataLP);
    smpGain = (float*)malloc(sizeof(float)*smpcount*channels);
    //calculates gain for each sample in every channel
    printf("calculating gain...\n");

    for(int ch=0; ch<channels; ch++)
    {
        for(int i=0; i<peaks[ch].size(); i++)
        {
            float peak_orig = (float)peaks[ch][i].value;
            float peak_proc = std::min(1.0f, peak_orig*gain);
            float gain_corrected = peak_proc/peak_orig;
            for(int j=peaks[ch][i].a; j<=peaks[ch][i].b; j++)
            {
                smpGain[j*channels+ch] = gain_corrected;
            }
        }
    }

	//smooths out the smpGain array using 3 passes of box blur in order to avoid clicks.
	///TODO: find a better approach

    printf("smoothing out the gain function: pass ");
    for(int bbp=0; bbp<boxblurpasses; bbp++)
    {
        printf("%d.. ", bbp);
        if(boxblurradius == 0)
            continue;
        float* avg = (float*)malloc(smpcount*channels*sizeof(float));
        for(int ch=0; ch<channels; ch++)
        {
            int el = 0;
            float average = 0;
            for(int j=0; j<boxblurradius; j++)
            {
                average += smpGain[j*channels+ch];
                el++;
            }
            for(int i=0; i<smpcount; i++)
            {
                int idx_start = i-boxblurradius-1;
                int idx_end = i+boxblurradius;
                if(idx_start >= 0)
                {
                    average -= smpGain[idx_start*channels+ch];
                    el--;
                }
                if(idx_end < smpcount)
                {
                    average += smpGain[idx_end*channels+ch];
                    el++;
                }
                avg[i*channels+ch] = average / float(el);
            }
        }
        memcpy(smpGain, avg, smpcount*channels*sizeof(float));
        free(avg);
    }
    puts("");

    aDataOut = (float*)malloc(smpcount*channels*sizeof(float));

    printf("applying gain...\n");

    for(int ch=0; ch<channels; ch++)
    {
        for(int i=0; i<smpcount; i++)
        {
            aDataOut[i*channels+ch] = aData[i*channels+ch] * smpGain[i*channels+ch] * postgain;
        }
    }

    printf("saving file...\n");

    free(aData);
    free(smpGain);

	aDataOutRaw = malloc(adSize);

    if(bitspersmp == 8)
    {
        uint8_t* cp = (uint8_t*)aDataOutRaw;

        for(int ch=0; ch<channels; ch++)
        {
            for(int i=0; i<smpcount; i++)
            {
                int32_t smp = (float(aDataOut[i*channels+ch]+1.0)*127.5);
                smp = i32max(0, i32min(255, smp));
                cp[i*channels+ch] = (uint8_t)smp;
            }
        }
    }
    if(bitspersmp == 16 || bitspersmp == 24)
    {
        if(bitspersmp == 24) //fuck you, gonna write in 16 anyway
            *(int16_t*)(&hdr_in[34]) = 16;
        int16_t* cp = (int16_t*)aDataOutRaw;
        for(int ch=0; ch<channels; ch++)
        {
            for(int i=0; i<smpcount; i++)
            {
                int32_t smp = aDataOut[i*channels+ch]*32765.0;
                smp = i32max(-32767, i32min(32766, smp));
                cp[i*channels+ch] = (int16_t)smp;
            }
        }
    }
    if(bitspersmp == 32)
    {
        memcpy(aDataOutRaw, aDataOut, adSize);
    }

    af_out.write(hdr_in, 44);
    af_out.write((char*)aDataOutRaw, adSize);
    af_in.close();
    //af_in_lp.close();
    af_out.close();

    return 0;
}
