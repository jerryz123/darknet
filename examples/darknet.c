#include "darknet.h"
#include "blas.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_attention(int argc, char **argv);
extern void run_regressor(int argc, char **argv);
extern void run_segmenter(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);
extern void run_lsd(int argc, char **argv);

void average(int argc, char *argv[])
{
   
}

long numops(network *net)
{
    int i;
    long ops = 0;

    return ops;
}

void speed(char *cfgfile, int tics)
{

}

void operations(char *cfgfile)
{
    
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
   
}

void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l)
{
   
}
void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
   
}

void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
   
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
   
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
   
}

layer normalize_layer(layer l, int n)
{

}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
 
}

void statistics_net(char *cfgfile, char *weightfile)
{
 
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{

}

void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix)
{

}

void visualize(char *cfgfile, char *weightfile)
{

}


void hexdump(const void* data, size_t size) {
	char ascii[17];
	size_t i, j;
	ascii[16] = '\0';
	for (i = 0; i < size; ++i) {
		printf("%02X ", ((unsigned char*)data)[i]);
		if (((unsigned char*)data)[i] >= ' ' && ((unsigned char*)data)[i] <= '~') {
			ascii[i % 16] = ((unsigned char*)data)[i];
		} else {
			ascii[i % 16] = '.';
		}
		if ((i+1) % 8 == 0 || i+1 == size) {
			printf(" ");
			if ((i+1) % 16 == 0) {
				printf("|  %s \n", ascii);
			} else if (i+1 == size) {
				ascii[(i+1) % 16] = '\0';
				if ((i+1) % 16 <= 8) {
					printf(" ");
				}
				for (j = (i+1) % 16; j < 16; ++j) {
					printf("   ");
				}
				printf("|  %s \n", ascii);
			}
		}
	}
        printf("---\n");
}

int main(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = -1;

   
    
    
    run_detector(argc, argv);
    return 0;
}

