'''
Created on May 14, 2013
Vision forward model for AoMR Shape grammar
Creates 3D scene according to given shape representation 
and uses VTK to render 3D scene to 2D image
@author: gerdogan
'''
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import scipy.misc

# Part positions in actual objects (given in meters)
actual_positions = {'Bottom0': [0.0, 0.0, -0.02280], 'Bottom1': [0.0, 0.0, -0.02280],
                    'Front0': [0.04, 0.0, 0.0], 'Front1': [0.04, 0.0, 0.0], 
                    'Ear0': [-0.0266666, 0.0, 0.0304], 'Ear1': [-0.0266666, 0.0, 0.0304], 
                    'Top0': [-0.04, 0.0, 0.0228], 'Top1': [-0.04, 0.0, 0.0228],
                    'Body': [0, 0, 0]}


class VisionForwardModel():
    """
    Vision forward model for AoMR Shape grammar
    Creates 3D scene according to given shape representation 
    and uses VTK to render 3D scene to 2D image
    """
    parts = ['Body', 'Bottom0', 'Bottom1', 'Front0', 'Front1', 'Top0', 'Top1', 'Ear0', 'Ear1']
    models_folder = '/home/goker/Dropbox/Code/Python/AoMRShapeGrammar/models/'
    view_camera_pos = (.16, -.16, .16) #canonical view
    camera_pos = [(.18, 0.0, 0.0), (0.0, -.30, 0.0), (0.0, 0.0, .30)] 
    camera_up = [(0, 0, 1), (0, 0, 1), (0, 1, 0)] # upward direction for each viewpoint
    render_size = (600, 600)
    save_image_size = (600, 600)
    def __init__(self, body_fixed=True):
        """
        body_fixed: if true Body part is automatically placed at origin
        """
        self.body_fixed = body_fixed
        # vtk objects for rendering
        self.vtkrenderer = vtk.vtkRenderer()
        
        self.viewpoint_count = len(self.camera_pos)
        self.vtkcamera = [] # one camera for each viewpoint
        for pos, up in zip(self.camera_pos, self.camera_up):
            camera = vtk.vtkCamera()
            camera.SetPosition(pos)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(up)
            self.vtkcamera.append(camera)
        
        self.vtkviewcamera = vtk.vtkCamera();
        self.vtkviewcamera.SetPosition(self.view_camera_pos);
        self.vtkviewcamera.SetFocalPoint(0, 0, 0);
        self.vtkviewcamera.SetViewUp(self.camera_up[0])
        
        self.vtkrender_window = vtk.vtkRenderWindow()
        self.vtkrender_window.AddRenderer(self.vtkrenderer)
        self.vtkrender_window.SetSize(self.render_size)
        self.vtkrender_window_interactor = vtk.vtkRenderWindowInteractor()
        self.vtkrender_window_interactor.SetRenderWindow(self.vtkrender_window)
        
        # vtk objects for reading, and rendering object parts
        self.vtkreader = {}
        self.vtkpolydata = {}
        self.vtkmapper = {}
        # read each part from its stl file
        for part in self.parts:
            self.vtkreader[part] = vtk.vtkSTLReader()
            self.vtkreader[part].SetFileName(self.models_folder + part + '.stl')
            self.vtkpolydata[part] = self.vtkreader[part].GetOutput()
            self.vtkmapper[part] = vtk.vtkPolyDataMapper()
            self.vtkmapper[part].SetInput(self.vtkpolydata[part])
        if self.body_fixed:
            # actor for body part (every object has part named body at origin)
            self.vtkbodyactor = vtk.vtkActor()
            self.vtkbodyactor.SetMapper(self.vtkmapper['Body'])
            self.vtkbodyactor.SetPosition(0, 0, 0)
            
    
    def render(self, *args):
        """
        Construct the 3D object from state and render it.
        Returns numpy array with size number of viewpoints x self.render_size
        """
        # called with ShapeState instance
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        img_arr = np.zeros((self.viewpoint_count, self.render_size[0], self.render_size[1]))
        self._build_scene(*parts_positions)
        for i, camera in enumerate(self.vtkcamera):
            self.vtkrenderer.SetActiveCamera(camera);
            barr = self._render_window_to2D()
            img_arr[i, :, :] = barr
        
        return img_arr
    
    def _render_window_to2D(self):
        """
        Renders the window to 2D black and white image
        Called from render function for each viewpoint
        """
        self.vtkrender_window.Render()
        self.vtkwin_im = vtk.vtkWindowToImageFilter()
        self.vtkwin_im.SetInput(self.vtkrender_window)
        self.vtkwin_im.Update()
        vtk_image = self.vtkwin_im.GetOutput()
        height, width, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
        # convert RGB image to grayscale image 
        arr = np.sum(arr, 2) / 3.0
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return arr
    
    def _build_scene(self, parts, positions):
        """
        Places parts to positions and adds them to scene
        Returns vtkRenderer
        """
        # clear scene
        self.vtkrenderer.RemoveAllViewProps()
        self.vtkrenderer.Clear()
        if self.body_fixed:
            # add body
            self.vtkrenderer.AddActor(self.vtkbodyactor)
        for part, position in zip(parts, positions):
            actor = vtk.vtkActor()
            actor.SetMapper(self.vtkmapper[part])
            actor.SetPosition(position)
            self.vtkrenderer.AddActor(actor)
                
    def _view(self, *args):
        """
        Views object in window
        Used for development and testing purposes
        """
        # called with ShapeState instance
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        self._build_scene(*parts_positions)
        self.vtkrenderer.SetActiveCamera(self.vtkviewcamera);
        self.vtkrender_window.Render()
        self.vtkrender_window_interactor.Start()
    
    def save_image(self, filename, *args):
        """
        Save image of object from canonical view to disk
        """
        # called with ShapeState instance
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        # img_arr = np.zeros((self.save_image_size[0], self.save_image_size[1]))
        self.vtkrender_window.SetSize(self.save_image_size)
        self.vtkrenderer.SetActiveCamera(self.vtkviewcamera);
        self._build_scene(*parts_positions)
        barr = self._render_window_to2D()
        self.vtkrender_window.SetSize(self.render_size)
        scipy.misc.imsave(filename, np.flipud(barr))
    
        
if __name__ == '__main__':
    forward_model = VisionForwardModel()
    parts = ['Bottom0', 'Front0', 'Top0', 'Ear0']
    positions = []
    for part in parts:
        positions.append(actual_positions[part])
    forward_model.save_image('test_vfm.png', parts, positions)     
    """
    #---------------------------------------------------------
    objects = [['Bottom0', 'Front0', 'Top0', 'Ear0'], ['Bottom0', 'Front0', 'Top0', 'Ear1'],
                ['Bottom0', 'Front0', 'Top1', 'Ear0'], ['Bottom0', 'Front0', 'Top1', 'Ear1'],
                ['Bottom0', 'Front1', 'Top0', 'Ear0'], ['Bottom0', 'Front1', 'Top0', 'Ear1'],
                ['Bottom0', 'Front1', 'Top1', 'Ear0'], ['Bottom0', 'Front1', 'Top1', 'Ear1'],
                ['Bottom1', 'Front0', 'Top0', 'Ear0'], ['Bottom1', 'Front0', 'Top0', 'Ear1'],
                ['Bottom1', 'Front0', 'Top1', 'Ear0'], ['Bottom1', 'Front0', 'Top1', 'Ear1'],
                ['Bottom1', 'Front1', 'Top0', 'Ear0'], ['Bottom1', 'Front1', 'Top0', 'Ear1'],
                ['Bottom1', 'Front1', 'Top1', 'Ear0'], ['Bottom1', 'Front1', 'Top1', 'Ear1']]
    
    # rotate camera around vertical axis and generate images
    positions = []
    view_angle = 0
    angle_step = 1
    camera_distance = 0.17 * np.sqrt(2)
    camera_z = .17

    for object_id, object in enumerate(objects):
        del positions[:]
        for part in object:
            positions.append(actual_positions[part])
            
        for i in range(360):
            view_angle += angle_step
            view_angle_radian = np.pi * (view_angle / 180.0)
            camera_x = camera_distance * np.sin(view_angle_radian)
            camera_y = camera_distance * np.cos(view_angle_radian)
            forward_model.vtkviewcamera.SetPosition((camera_x, camera_y, camera_z))
            #forward_model._view(parts, positions)
            forward_model.save_image('./data/rotated_fmri/' + repr(object_id) + '_' + repr(i) + '.png', object, positions)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # generate data for each object
#     objects = [['Bottom0', 'Front0', 'Top0', 'Ear0'], ['Bottom0', 'Front0', 'Top0', 'Ear1'],
#                ['Bottom0', 'Front0', 'Top1', 'Ear0'], ['Bottom0', 'Front0', 'Top1', 'Ear1'],
#                ['Bottom0', 'Front1', 'Top0', 'Ear0'], ['Bottom0', 'Front1', 'Top0', 'Ear1'],
#                ['Bottom0', 'Front1', 'Top1', 'Ear0'], ['Bottom0', 'Front1', 'Top1', 'Ear1'],
#                ['Bottom1', 'Front0', 'Top0', 'Ear0'], ['Bottom1', 'Front0', 'Top0', 'Ear1'],
#                ['Bottom1', 'Front0', 'Top1', 'Ear0'], ['Bottom1', 'Front0', 'Top1', 'Ear1'],
#                ['Bottom1', 'Front1', 'Top0', 'Ear0'], ['Bottom1', 'Front1', 'Top0', 'Ear1'],
#                ['Bottom1', 'Front1', 'Top1', 'Ear0'], ['Bottom1', 'Front1', 'Top1', 'Ear1']]
#      
#     forward_model = VisionForwardModel()
#      
#     positions = []
#     for i, object in enumerate(objects):
#         del positions[:]
#         for part in object:
#             positions.append(actual_positions[part])
#              
#         render = forward_model.render(object, positions)
#         scipy.misc.imsave(repr(i+1) + '_1.png', render[0,:,:])
#         scipy.misc.imsave(repr(i+1) + '_2.png', render[1,:,:])
#         scipy.misc.imsave(repr(i+1) + '_3.png', render[2,:,:])
#         np.save(repr(i+1) + '.npy', render)
#      
#     # ---------------------------------------------------------
    """
