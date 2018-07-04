#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <mvnc.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
  
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#include "fp16.h"
using namespace std;
using namespace cv;
#define NAME_SIZE 100
#define GRAPH_FILE_NAME "../graph"
#define IMG_FILE_NAME "/home/lq/Camera Roll/2.jpg"
typedef unsigned short half;
const int networkDim=300;
float networkMean[]={0.007843*127.0, 0.007843*127.0, 0.007843*127.0};
int image_width=396;
int image_heigh=448;

char* LABELS[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike","'person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};


int  OverlayOnImage(char* img_path ,int* obj_info);
void draw_rectangle(char *img_path, int* loc);
void overlayOnImage(char* img_path ,float* obj_info);

void *LoadFile(const char *path, unsigned int *length)
{
	FILE *fp;
	char *buf;

	fp = fopen(path, "rb");
	if(fp == NULL)
		return 0;
	fseek(fp, 0, SEEK_END);
	*length = ftell(fp);
	rewind(fp);
	if(!(buf = (char*) malloc(*length)))
	{
		fclose(fp);
		return 0;
	}
	if(fread(buf, 1, *length, fp) != *length)
	{
		fclose(fp);
		free(buf);
		return 0;
	}
	fclose(fp);
	return buf;
}


half *LoadImage(const char *path, int reqsize, float *mean)
{
	int width, height, cp, i;
	unsigned char *img, *imgresized;
	float *imgfp32;
	half *imgfp16;

	img = stbi_load(path, &width, &height, &cp, 3);
       
        
	if(!img)
	{
		printf("The picture %s could not be loaded\n", path);
		return 0;
	}
	imgresized = (unsigned char*) malloc(3*reqsize*reqsize);
	if(!imgresized)
	{
		free(img);
		perror("malloc");
		return 0;
	}
	stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);
	free(img);
	imgfp32 = (float*) malloc(sizeof(*imgfp32) * reqsize * reqsize * 3);
	if(!imgfp32)
	{
		free(imgresized);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize * reqsize * 3; i++)
		imgfp32[i] = imgresized[i];
	free(imgresized);
	imgfp16 = (half*) malloc(sizeof(*imgfp16) * reqsize * reqsize * 3);
	if(!imgfp16)
	{
		free(imgfp32);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize*reqsize; i++)
	{
		float blue, green, red;
                blue = imgfp32[3*i+2]*0.007843;
                green = imgfp32[3*i+1]*0.007843;
                red = imgfp32[3*i+0]*0.007843;

                imgfp32[3*i+0] = blue-mean[0];
                imgfp32[3*i+1] = green-mean[1]; 
                imgfp32[3*i+2] = red-mean[2];

                // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
                //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
	}
	floattofp16((unsigned char *)imgfp16, imgfp32, 3*reqsize*reqsize);
	free(imgfp32);
	return imgfp16;
}


float* run_inference( int networkDim, void* graphHandle)
{
	
        half* imageBufFp16 = LoadImage(IMG_FILE_NAME, networkDim, networkMean);
  
        unsigned int lenBufFp16 = 3*networkDim*networkDim*sizeof(*imageBufFp16);

        // start the inference with mvncLoadTensor()
        mvncLoadTensor(graphHandle, imageBufFp16, lenBufFp16, NULL);

        void* resultData16;
        void* userParam;
        unsigned int lenResultData;
        mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
        
        int numResults = lenResultData / sizeof(half);
        float* output;
        output = (float*)malloc(numResults * sizeof(*output));
        fp16tofloat(output, (unsigned char*)resultData16, numResults);

        //for (int i=0; i<lenResultData; i++)
          //   printf("%f ", resultData32[i]);
	int num_valid_boxes=(int)output[0];
	int x1,y1,x2,y2;
//	printf("%d",num_valid_boxes);
	for(int box_index=0; box_index<num_valid_boxes; box_index++)
	{
		int base_index= 7+box_index*7;	
		if(!isfinite(output[base_index + 1])||!isfinite(output[base_index + 2])||!isfinite(output[base_index + 3])||!isfinite(output[base_index + 4])||!isfinite(output[base_index + 5])||!isfinite(output[base_index + 6]))
     			continue;

	
		x1=(int)(output[base_index+3]*image_width);//形状是图像到长宽，一会用cv2里的库函数来实现吧
		y1=(int)(output[base_index+4]*image_heigh);
		x2=(int)(output[base_index+5]*image_width);
		y2=(int)(output[base_index+6]*image_heigh);
	//	printf("imag_width:%d,imag_heigh:%d all",image_width, image_width);
		if(x1<0||y1<0||x2<0||y2<0)
		{
			printf("ignore this boxes");
			continue;
 		
		}

		if(x1>image_width||y1>image_heigh||x2>image_width||y2>image_heigh)
		{
			printf("box out of boundary, ignoring it");
			continue;
		}
		printf("%f,%f,%f,%f\n",output[base_index+3],output[base_index+4],output[base_index+5],output[base_index+6]);
  		printf("%d,%d,%d,%d\n",x1,y1,x2,y2);
//		char X1=x1+'0';	
//		char Y1=y1+'0';
//		char X2=x2+'0';
 // 		char Y2=y2+'0';
//		int class_index=(int)(output[base_index+ 1]);
//		int confidence=(int)(output[base_index+ 2]*100);
//		printf("class ID: %s,Confidence:%f,location: %d, %d, %d, %d",*LABELS[class_index],confidence,x1,y1,x2,y2);
	 //       float obj_info[6]={output[base_index+1],output[base_index+2],output[base_index+3],output[base_index+4],output[base_index+5],output[base_index+6]};
         	int obj_info[4]={x1,y1,x2,y2};
		OverlayOnImage(IMG_FILE_NAME, obj_info);


	}
	
        
        return output;

}



int  OverlayOnImage(char* img_path ,int* obj_info)
{

	draw_rectangle(img_path, obj_info);	
	return 0;
}

void overlayOnImage(char* img_path ,float* obj_info)
{
	int min_score_percent=20;
	int base_index=0;
	
	int class_id=(int) (obj_info[base_index+1]);
	int percentage=(int)(obj_info[base_index+2]*100);
	if(percentage<min_score_percent)
	{
		return ;
	}

	char* label_text= LABELS[class_id];
	char* score;
	//itoa(percentage, score, 10);
	//strcat(label_text,score);

	int x1=(int)(obj_info[base_index+3]*image_width);//形状是图像到长宽，一会用cv2里的库函数来实现吧
	int y1=(int)(obj_info[base_index+4]*image_heigh);
	int x2=(int)(obj_info[base_index+5]*image_width);
	int y2=(int)(obj_info[base_index+6]*image_heigh);

        printf("%d,%d,%d,%d",x1,y1,x2,y2);
        int loc[4]={x1,y1,x2,y2};
	draw_rectangle(img_path, loc);	


}


void draw_rectangle(char *img_path, int* loc)
{


 	Mat matImage = cv::imread(img_path,-1);

//        IplImage *iplImage = cvLoadImage(img_path,-1);


	//Rect(int a,int b,int c,int d)a,b为矩形的左上角坐标,c,d为矩形的长和宽


	cv::rectangle(matImage,Rect(loc[0],loc[1],loc[2],loc[3]),Scalar(255,128,0),1,1,0);

	

	imshow("matImage",matImage);

	waitKey(-1);
//	cvShowImage("IplImage",iplImage);

}

void test()
{

float a[4]={10,10,40,50};
overlayOnImage(IMG_FILE_NAME ,a);
	waitKey(-1);



}



int main(int arg, char** argv)
{

    mvncStatus retCode;
    void *deviceHandle;
    char devName[NAME_SIZE];

    retCode = mvncGetDeviceName(0, devName, NAME_SIZE);
    if (retCode != MVNC_OK)
    {   // failed to get device name, maybe none plugged in.
        printf("No NCS devices found\n");
        exit(-1);
    }

    retCode = mvncOpenDevice(devName, &deviceHandle);
    if (retCode != MVNC_OK)
    {   // failed to open the device.  
        printf("Could not open NCS device\n");
        exit(-1);
    }


    // Now read in a graph file
    unsigned int graphFileLen;
    void* graphFileBuf = LoadFile(GRAPH_FILE_NAME, &graphFileLen);

    // allocate the graph
    void* ssd_mobilenet_graph;
    retCode = mvncAllocateGraph(deviceHandle, &ssd_mobilenet_graph, graphFileBuf, graphFileLen);

    float* result;
    result = run_inference(networkDim,ssd_mobilenet_graph);
    

 //   printf("%f\n",result[0]);

    //space release and decive close
    mvncDeallocateGraph(ssd_mobilenet_graph);
    ssd_mobilenet_graph = NULL;
    free(graphFileBuf);
    retCode = mvncCloseDevice(deviceHandle);
    deviceHandle = NULL;


}

