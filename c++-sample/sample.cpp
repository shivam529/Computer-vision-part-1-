//Link to the header file
#include "CImg.h"
#include <iostream>
#include <stdlib.h>

//Use the cimg namespace to access the functions easily
using namespace cimg_library;
using namespace std;

int main(){

	//Load an image (this one happens to be grayscale)
	CImg<double> image("example.jpg");

	//Check its width, height, and number of color channels
	cout << "Image is " << image.width() << " pixels wide." << endl;
	cout << "Image is " << image.height() << " pixels high." << endl;
	cout << "Image has " << image.spectrum() << " color channel(s)." << endl;

	//Images are indexed by (X,Y,Z,C) where X and Y are location in image, 
	//Z is which image in a composite (always zero for our purposes), and
	//the color channel C. For a color images 0=Red, 1=Green, 2=Blue.
	cout << "Pixel value is " << image(10,10,0,0) << endl;	
	image(10,10,0,0) = 20;
	cout << "New pixel value is " << image(10,10,0,0) << endl;	
	
	//Create a new blank color image the same size as the input
	CImg<double> color_image(image.width(), image.height(), 1, 3);

	//Loops over the new color image and fills in any area that was white in the 
	//first grayscale image we loaded with random colors!
	for(int x=0; x < color_image.width(); x++){
		for(int y=0; y < color_image.height(); y++){
			
			if(image(x,y,0,0) > 200){		
				color_image(x,y,0,0) = rand()/(double)RAND_MAX*256; //Random red value
				color_image(x,y,0,1) = rand()/(double)RAND_MAX*256; //Random green value
				color_image(x,y,0,2) = rand()/(double)RAND_MAX*256; //Random blue value
			}else{
				color_image(x,y,0,0) = 0; //Black value for the red channel
				color_image(x,y,0,1) = 0; //Black value for the green channel
				color_image(x,y,0,2) = 0; //Black value for the blue channel
			}
		}
	}	

	//Save the image
	color_image.save("output.jpg");

	return 0;
}





