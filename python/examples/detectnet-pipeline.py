#!/usr/bin/python
#

import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live gStreamer stream using an object detection DNN.",
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="facenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.8, help="minimum detection threshold to use (default is 0.8)")
parser.add_argument("--pipeline", type=str, default="", help="gStreamer pipeline string")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--depth", type=int, default=12, help="desired pixel depth of camera stream (default is 12)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create the camera and display
camera = jetson.utils.gstPipeline(opt.pipeline, opt.width, opt.height, opt.depth)
display = jetson.utils.glDisplay()

# process frames until user exits
while display.IsOpen():
	# capture the image
	img, width, height = camera.CaptureRGBA()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, width, height, opt.overlay)

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)

	# render the image
	display.RenderOnce(img, width, height)

	# update the title bar
	display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()
