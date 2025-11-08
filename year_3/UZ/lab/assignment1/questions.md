# Questions

## Exercise 1

### Why would you use different color maps?
They are used for different purposes.

grayscale - intuitive for single channel data (black to white) 
viridis - purple to yellow colors, that help highlight intensity variations that may be harder to see on pure grayscale
other colormaps - can emphasize other features of the image

### How is inverting a gray-scale value defined?
The value of pixel P is defined via the following formula:
P_inverted = 255 - P;

## Exercise 2

### The histograms are usually normalized by dividing the result by the sum of all cells. Why is that?
By that we convert absolute counts to relative frequencies / probabilities.

## Exercise 3

### Based on the results, which order of erosion and dilation operations produces opening and which closing?
Opening = Erosion followed by Dilation
Closing = Dilation followed by Erosion

### Why is the background included in the mask and not the object? How would you fix that in general? (just inverting the mask if necessary doesn’t count)
The background is in the mask because it’s brighter than the object and above the 0.3 threshold.

Fixes: ehnance contrast or preprocess the object so it stands out more than the background
