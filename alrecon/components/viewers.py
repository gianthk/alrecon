import logging
from os import system

class viewers:
    def __init__(self):
        self.napari_available = True

        try:
            import napari
        except:
            self.napari_available = False

    def napariView(self, image):
        if self.napari_available:
            import napari
            viewer = napari.view_image(image)
        else:
            logging.ERROR("Napari module not available.")

    def imagejView(self, ImageJ_exe_stack, image_dir, image_name=None):
        if image_name is not None:
            # system(ImageJ_exe_stack + image_dir + '/{:04.2f}'.format(ar.COR_range.value[0]) + '.tiff &')
            system(ImageJ_exe_stack + image_dir + image_name + '.tiff &')
        else:
            system(ImageJ_exe_stack + image_dir + '/slice.tiff &')
