import numpy as np 


class Conv3x3:
    # conv layer with 3*3 filters or kernels or receptive fields

    def __init__(self, num_filters) :
        
        self.num_filters = num_filters

        self.filters = np.random.randn(num_filters, 3, 3) / 9

        # dividing by 9 is Xavier's Initializatin, where you dont want the initial
        # weights to lie in the dead zone by wither being too small or way too big

    
    def iterate_regions(self, image) :
        '''
        Generated all 3x3 image regions using valid padding, that is 1px padding
    
        params :
        ----------
        image : a 2d numpy array
        '''
        
        h, w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                
                im_region = image[i:(i+3), j:(j+3)] # slicing 3x3 sized regions
                
                yield im_region, i, j # returns a generator object, which we can then iterate over
    

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions( h, w, num_filters )
        
        params :
        ----------
        input: a 2d numpy array
        
        '''

        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis = (1, 2))

        return output