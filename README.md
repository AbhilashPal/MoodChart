# MoodChart
Using a CNN to detect mood and plot it in a Bar Chart.

Usage :
--
 To use simply clone the repo and run the mood.py file using the command line.
 A CV2 window will open. Press v(caps lock off) to click your picture and run 
 the analysis on the image. Make sure you click the image after cv2 has detected your
 face(blue square over face)

Model:
--
 The model is trained using a CNN network created using keras and the following dataset :
 https://github.com/muxspace/facial_expressions
 
 Trained using a GTX 1050Ti on a local system. The training files can be found in here :
 https://github.com/AbhilashPal/Blissify.ai/tree/master/model
