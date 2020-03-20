/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstPipeline.h"
#include "glDisplay.h"

#include "detectNet.h"
#include "commandLine.h"

#include <signal.h>


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: detectnet-pipeline [-h] [--network NETWORK] [--threshold THRESHOLD]\n");
	printf("                        [--pipeline GST-PIPELINE] [--width WIDTH] [--height HEIGHT] [--depth DEPTH]\n\n");
	printf("Locate objects in a gst stream using an object detection DNN.\n\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network NETWORK pre-trained model to load (see below for options)\n");
	printf("  --overlay OVERLAY detection overlay flags (e.g. --overlay=box,labels,conf)\n");
	printf("                    valid combinations are:  'box', 'labels', 'conf', 'none'\n");
     printf("  --alpha ALPHA     overlay alpha blending value, range 0-255 (default: 120)\n");
	printf("  --pipeline GST-PIPELINE gst-pipline as string, e.g.:,\n");
	printf("                    rtspsrc location=rtsp://user:pw@192.168.0.170/Streaming/Channels/1 ! queue ! rtph264depay ! h264parse ! queue ! omxh264dec ! appsink name=mysink.\n");
	printf("  --width WIDTH     desired width of pipeline stream (default is 1280 pixels)\n");
	printf("  --height HEIGHT   desired height of pipeline stream (default is 720 pixels)\n");
	printf("  --threshold VALUE minimum threshold for detection (default is 0.5)\n\n");

	printf("%s\n", detectNet::Usage());

	return 0;
}

int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the pipeline
	 */
    gstPipeline* pipeline = gstPipeline::Create(cmdLine.GetString("pipeline"),
            cmdLine.GetInt("width", gstPipeline::DefaultWidth),
            cmdLine.GetInt("height", gstPipeline::DefaultHeight),
            gstPipeline::DefaultDepth);

	if( !pipeline )
	{
		printf("\ndetectnet-pipeline:  failed to initialize pipeline device\n");
		return 0;
	}
	
	printf("\ndetectnet-pipeline:  successfully initialized pipeline device\n");
	printf("    width:  %u\n", pipeline->GetWidth());
	printf("   height:  %u\n", pipeline->GetHeight());
	printf("    depth:  %u (bpp)\n\n", pipeline->GetPixelDepth());
	

	/*
	 * create detection network
	 */
	detectNet* net = detectNet::Create(argc, argv);
	
	if( !net )
	{
		printf("detectnet-pipeline:   failed to load detectNet model\n");
		return 0;
	}

	// parse overlay flags
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();

	if( !display ) 
		printf("detectnet-pipeline:  failed to create openGL display\n");


    display->SetViewport(10, 10, pipeline->GetWidth(), pipeline->GetHeight());

	/*
	 * start streaming
	 */
	if( !pipeline->Open() )
	{
		printf("detectnet-pipeline:  failed to open pipeline for streaming\n");
		return 0;
	}

	printf("detectnet-pipeline:  pipeline open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		// capture RGBA image
		float* imgRGBA = NULL;
		
		if( !pipeline->CaptureRGBA(&imgRGBA, 1000) )
			printf("detectnet-pipeline:  failed to capture RGBA image from pipeline\n");

		// detect objects in the frame
		detectNet::Detection* detections = NULL;
	
		const int numDetections = net->Detect(imgRGBA, pipeline->GetWidth(), pipeline->GetHeight(), &detections, overlayFlags);
		
		if( numDetections > 0 )
		{
			printf("%i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
				printf("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				printf("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
			}
		}	

		// update display
		if( display != NULL )
		{
			// render the image
			display->RenderOnce(imgRGBA, pipeline->GetWidth(), pipeline->GetHeight());

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			display->SetTitle(str);

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}

		// print out timing info
		//net->PrintProfilerTimes();
	}
	

	/*
	 * destroy resources
	 */
	printf("detectnet-pipeline:  shutting down...\n");
	
	SAFE_DELETE(pipeline);
	SAFE_DELETE(display);
	SAFE_DELETE(net);

	printf("detectnet-pipeline:  shutdown complete.\n");
	return 0;
}

