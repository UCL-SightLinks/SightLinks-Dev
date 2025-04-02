# Feature Extraction Experimental Methods - Our method actually does not signficiantly benefit from 
# common preprocessing techniques, so we created a set of more uncommon/ niche methods as well as some
#  other untested methods to experiment with which features best represent the crosswalk features.

# This hopefully should be of great use to anyone trying to train their own specialised classifier using 
# our pipeline - hopefully one of these works for you!

# All these methods work on a numpy array - if you want to use them convert your image to numpy first
# Most of the methods assume the channels of the image are at the end

import numpy as np
from scipy.signal import convolve2d
import skimage.feature as skf

from scipy.ndimage import maximum_filter
from collections import deque



# Was required for keeping several of the methods optimised.
def matrixConvolution(image, kernel):
    if len(image.shape) < 3:
        raise ValueError("Input image must have 3 dimensions (h, w, c)")
    convolvedImage = np.zeros_like(image)

    # Again, this assumes the channels are placed at the end of the image
    for c in range(image.shape[2]):
        convolvedImage[..., c] = convolve2d(image[..., c], kernel, mode='same', boundary='wrap')

    return convolvedImage


# Different more traditional image processing methods, these may not be of particular use in our project, 
# but could be useful for other developers using this pipeline.
class ImageProcessingFeatures:
    def __init__(self):
        self.gaussianKernel = self.generateGaussianKernel(5, 1)

    # Converts an image to grayscale - but using binary thresholding of the averaged colour channels
    # Image in format [n, m, k] where k is the channels and n, m are the dimensions
    def binaryThresholding(self, image, threshold):

        averaged = np.mean(image, axis=2)
        thresholded = averaged > threshold

        return thresholded
    
    # A bit unintuitively named, but this takes an already grayscale image and applies binary thresholding to it
    def grayscaleBinaryThresholding(self, grayscaleImage, threshold):
        return grayscaleImage > threshold

    
    # Converts an image to grayscale - it has several possible schema it can use but defaults to the lightness method
    def grayscaleConversion(self, image, schema="lightness"):
        # https://tannerhelland.com/2011/10/01/grayscale-image-algorithm-vb6.html
        grayImage = np.zeros(np.shape(image)[:-1]) 

        if schema == "average":
            grayImage = np.mean(image, axis=2)

        if schema == "lightness":
            # (max(R, G, B) + min(R, G, B)) / 2 --> Sometimes called desaturation
            grayImage = (np.max(image, axis=2) + np.min(image, axis=2)) / 2

        if schema == "luma":
            # I assume RGB colour ordering in the image here -- feel free to overwrite in your implementation
            # (Red * 0.2126 + Green * 0.7152 + Blue * 0.0722)
            colourWeighting = np.array([0.2126, 0.7152, 0.0722])
            grayImage = np.dot(image[..., :3], colourWeighting)

        if schema == "decomposition":
            # This is a maximum decomposition - minumum decomposition can be implemented by just switching our for min...
            grayImage = np.max(image, axis=2)

        return grayImage


    # Sharpens the image by convolving a preset laplace kernel with it
    # My chosen kernel is a pretty standard one that takes into consideration the corners - the 8 adjacencies one
    # ​[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]
    def laplaceTransform(self, image):
        laplaceKernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        convolvedImage = None

        # RGB image
        if len(np.shape(image)) == 3:
            convolvedImage = matrixConvolution(image, laplaceKernel)

        # GrayScale image
        if len(np.shape(image)) == 2:
            convolvedImage = convolve2d(image, laplaceKernel, mode='same', boundary='wrap')

        return convolvedImage

    # Takes Grayscale images
    # https://en.wikipedia.org/wiki/Sobel_operator - naive implementation by me, can probably be massively improved
    # Basically takes the vertical and horizontal gradients using convolution with a sobel operator and combines them.
    def sobelConvolution(self, image):

        if len(np.shape(image)) >= 3:
            image = self.grayscaleConversion(image)

        # Gx
        verticalKernel = [[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]

        # Gy
        horizontalKernel = [[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]] 

        horizontalGrad = convolve2d(image, horizontalKernel, mode='same', boundary='wrap')
        verticalGrad = convolve2d(image, verticalKernel, mode='same', boundary='wrap')

        sobelConvolvedImage = np.sqrt(np.square(horizontalGrad) + np.square(verticalGrad))

        # You could go further to find the gradient direction by calculating angle with some trigonometry - atan2(Gy, Gx)
        return sobelConvolvedImage
    
    # Part of the calculations for canny edge detection, a tri-threshold operations. Requires greyscale images.
    # Classifies into strong edges and weak edges based on pixel intensity
    def doubleFiltering(self, image, weakThreshold=75, strongThreshold=200):
        weak, strong = 125, 255

        strong_edges = image >= strongThreshold
        weak_edges = (image >= weakThreshold) & (image < strongThreshold)
        
        result = np.zeros_like(image, dtype=np.uint8)
        result[strong_edges] = strong
        result[weak_edges] = weak
        
        return result, strong_edges, weak_edges
    
    # Technical name: edge tracking by hysteresis. This is an 'optimised' dequeue approach (to the best of my ability)
    def followEdges(self, weakEdges, strongEdges):
        h, w = np.shape(weakEdges)
        directions = [(-1, -1), (-1, 0), (0, -1), (1, -1), (-1, 1), (0, 0), (0, 1), (1, 0), (1, 1)]
        finalEdges = (strongEdges.copy() > 0) # must include strong edges, it's guaranteed, and we want a binary output.
        edgeQueue = deque(map(tuple, np.argwhere(strongEdges == 1)))
        
        while edgeQueue:
            y, x = edgeQueue.popleft()  
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                # Expand the branch if they're connected to a weak edge, else kill this branch
                if 0 <= ny < h and 0 <= nx < w and weakEdges[ny, nx] > 0:
                    weakEdges[ny, nx] = 0
                    finalEdges[ny, nx] = 1  
                    edgeQueue.append((ny, nx))
        
        return finalEdges


    # Much better than sobel in terms of accuracy and removing false edges produced due to noise, but slower.
    # https://en.wikipedia.org/wiki/Canny_edge_detector + a lot of chatGPT prompts
    def cannyEdgeDetection(self, image):
        # Ensures that the image is in greyscale so we don't have any issues.
        if (len(np.shape(image)) == 3):
            image = self.grayscaleConversion(image, schema="average")

        # Takes a greyscale image and returns a set of strong and weak edges (see wikipedia page)
        blurred = convolve2d(image, self.gaussianKernel, mode='same', boundary='wrap')
        gradientMagnitude = self.sobelConvolution(blurred)
        maxFiltered = maximum_filter(gradientMagnitude, size=3, mode='constant')
        suppressed = np.where(gradientMagnitude==maxFiltered, gradientMagnitude, 0)
        thresholded, strong, weak = self.doubleFiltering(suppressed)

        final_edges = self.followEdges(weakEdges=weak, strongEdges=strong)
        # We expand the definition of strong edges to include weak edges that are adjacent to strong edges

        return final_edges

    
    # Convolve this with the image to produce a Gaussian Blur effect 
    # In a real application you should precompute this, it's really unoptimised to recalculate it each time
    def generateGaussianKernel(self, size, sigma):
        center = size // 2
        kernel = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                sides = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[i, j] = np.exp(-(sides ** 2) / (2 * sigma ** 2))  # According to the formula

        return kernel / np.sum(kernel)


    # This should be applied to a grayscale image! 
    def differenceOfGaussians(self, image, size=5, sigmaOne=1, sigmaTwo=1.5, brightFeatureFocus=True):
        # This by default uses a 5x5 kernel with sigmaOne = 1 and sigmaTwo = 2. These are not finetuned values but follow the general
        # principle that sigma should not be greater than approx. 3*dims. Be careful to consider the relationship between the dimension
        # and sigma values (and between the two sigmas themselves) to preserve the gaussian property!

        if len(np.shape(image)) >= 3:
            image = self.grayscaleConversion(image)

        kernelOne, kernelTwo = self.generateGaussianKernel(size, sigmaOne), self.generateGaussianKernel(size, sigmaTwo)

        primaryImage = convolve2d(image, kernelOne, mode='same', boundary='wrap')
        backgroundImage = convolve2d(image, kernelTwo, mode='same', boundary='wrap')

        diffOfGaussians = primaryImage - backgroundImage
        
        # Enhances bright feature edges (our focus in crosswalks), but this is a thing that might vary based on your focus. Can be disabled!
        if brightFeatureFocus:
            diffOfGaussians = np.maximum(diffOfGaussians, 0)

        return diffOfGaussians


# Different feature extraction methods that attempt to quantify the complexity in an image. 
# Generally seperated into local complexity per region, and complexity of an image as a whole.
# Intended to potentially detect the occlusion in an image, e.g. more treees --> more complexity hopefully.
class ComplexityFeatures:
    def __init__(self):
        self.imgProc = ImageProcessingFeatures()
        pass

    # Definition: Mathematical measure of how complex an image or pattern is
    # In our case (Grayscale image), it measures how much the detail in an image changes with the scale it is perceived at.
    # The particular method we will the basic box-counting method
    # Our definition of a useful feature map is a binary-thresholded line detection method using difference of Gaussians
    def fractalDimension(self, image, minimumBoxSize=2, imageStructureThreshold=0.9):

        if len(image.shape) > 2:
            image = self.imgProc.grayscaleConversion(image, schema="average")

        image = self.imgProc.sobelConvolution(image) # Could also go for DoG or Laplace too - most edge detectors work
        image = np.uint8(255 * (image - np.min(image)) / (np.max(image) - np.min(image)))
        image = self.imgProc.grayscaleBinaryThresholding(image, imageStructureThreshold) 

        N, M = image.shape

        startingBoxSize = min(N, M) // 4
        boxSizes = [s for s in range(minimumBoxSize, startingBoxSize + 1, 2) if min(N, M) % s == 0]
        logSizes, logCounts = [], []

        for boxSize in boxSizes:
            numBoxes = 0

            for i in range(0, N, boxSize):
                for j in range(0, M, boxSize):
                    if np.any(image[i:i+boxSize, j:j+boxSize]):  # Check if box contains a 1 (part of the image structure)
                        numBoxes += 1
                        
            if numBoxes > 0:
                logSizes.append(np.log(1.0 / boxSize)) 
                logCounts.append(np.log(numBoxes))
        
        slope, _ = np.polyfit(logSizes, logCounts, 1)
        # Finds the relationship of the sizes and the image structure contained as the image scale decreases

        return slope

    # Not implemented - left as a suggest for a future improvement in method.
    def waveletBasedFractalTransform(self, image):
        pass


# A set of feature extraction methods inspired by texture analysis methods
# just a wild throw in the dark, not sure how useful these could be as a feature
# Many of these should be applied in sliding window approaches or in regions, or to the whole image if you have a feature vector
class TextureFeatures:
    def __init__(self):
        self.imgProc = ImageProcessingFeatures()
        pass

    def lbpCompare(self, threshold, value):
        return 0 if value < threshold else 1

    # A matter of personal preference, but this is a spiral concatenation for the local binary pattern signature generation
    # You could do row by row, but this is my preferred method. Feel free to overwrite, it shouldn't make a difference as long as you're consistent
    def spiral_concatenation(self, vals, dims):
        summed = []
        dims_x, dims_y = dims
        cur_dir = 0
        cur_x, cur_y = 0, 0
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while dims_x > 0 and dims_y > 0:
            for _ in range(dims_x):
                summed.append(str(vals[cur_y * dims_y + cur_x]))
                cur_y, cur_x = cur_y + dirs[cur_dir][0], cur_x + dirs[cur_dir][1]
            cur_dir = (cur_dir + 1) % 4
            dims_x, dims_y = dims_x - 1, dims_y - 1

        return "".join(summed)

    # Basically captures the local binary changes in texture in an image - a potentially useful feature for our crosswalk detector that
    # works on a very similar principle with the local regions of interest found by the first.
    # It generates a signature for each local region that can be used to compare them quite easily in applications like texture analysis.
    # CAREFUL - THIS IS A STRING FEATURE, NOT A NUMERICAL ONE
    def localBinaryPattern(self, image, dims):
        image = self.imgProc.grayscaleConversion(image)
        imgWidth, imgLength = len(image[0]), len(image)
        edge = dims // 2 
        lbpList = []
        for row in range(edge, imgLength - edge):
            for pixel in range(edge, imgWidth - edge):
                neighborhood = image[row - edge: row + edge + 1, pixel - edge: pixel + edge + 1]
                
                centralPixel = image[row][pixel]
                binaryVals = [self.lbpCompare(centralPixel, val) for val in neighborhood.flatten()]

                lbpSignature = self.spiral_concatenation(binaryVals, (dims, dims))
                lbpList.append(lbpSignature)

        return lbpList

    # https://medium.com/@girishajmera/feature-extraction-of-images-using-glcm-gray-level-cooccurrence-matrix-e4bda8729498
    # Link above explains the function quite well and succinctly. This takes in both coloured (n, m, k) and grayscale arrays (n, m).
    # Captures the spatial relationships between neighbouring gray levels/ Intensities
    def grayLevelCoOccurrenceMatrix(self, image, pixelOffset=5, preserveMatrix=False):
        if not (len(np.shape(image)) == 2):
            # P.S - this assumes that RGB and [n, m, k] formats are followed. Can throw errors otherwise.
            image = self.imgProc.grayscaleConversion(image, schema="average")

        transformedArray = np.uint8(255 * (image - np.min(image)) / (np.max(image) - np.min(image)))

        distances = [pixelOffset]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        # A bit of a cop out, but this just does it all for us. The metrics you choose to extract from this depend.
        glcm = skf.graycomatrix(transformedArray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # This is a set of features that I personally thought might be useful for the crosswalks - but there are many other extractable features
        contrast = skf.graycoprops(glcm, 'contrast')
        energy = skf.graycoprops(glcm, 'energy')
        homogeneity = skf.graycoprops(glcm, 'homogeneity')
        correlation = skf.graycoprops(glcm, 'correlation')

        # Single value metrics for the image
        metrics = (np.mean(contrast.flatten()), np.mean(energy.flatten()), np.mean(homogeneity.flatten()), np.mean(correlation.flatten()))

        if preserveMatrix:
            return metrics, (contrast, energy, homogeneity, correlation)
        
        return metrics


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Random Example
    image = Image.open("zebra_annotations/zebra_images/9186868439.jpg")

    image_array = np.array(image)


    textFet = TextureFeatures()
    compFet = ComplexityFeatures()
    imgProc=  ImageProcessingFeatures()

    print(imgProc.laplaceTransform(image_array))
    print(imgProc.sobelConvolution(image_array))
    print(imgProc.cannyEdgeDetection(image_array))
    print(imgProc.differenceOfGaussians(image_array))

    print(compFet.fractalDimension(image_array))
    print(textFet.localBinaryPattern(image_array, 2))
    print(textFet.grayLevelCoOccurrenceMatrix(image_array))