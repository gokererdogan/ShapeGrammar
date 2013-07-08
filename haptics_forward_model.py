'''
Created on May 23, 2013
Haptics forward model for AoMR Shape grammar
Creates 3D scene according to given shape representation 
and uses VTK to save 3D scene as VRML file, then imports
this file to GraspIt and gets joint angles of human hand 
from GraspIt when hand auto grasps the object
@author: gerdogan
'''
import vtk
import socket
import numpy as np
import scipy.misc
import matplotlib.pyplot as pl

# Part positions in actual objects (given in meters)
actual_positions = {'Bottom0': [0.0, 0.0, -0.02280], 'Bottom1': [0.0, 0.0, -0.02280],
                    'Front0': [0.04, 0.0, 0.0], 'Front1': [0.04, 0.0, 0.0], 
                    'Ear0': [-0.0266666, 0.0, 0.0304], 'Ear1': [-0.0266666, 0.0, 0.0304], 
                    'Top0': [-0.04, 0.0, 0.0228], 'Top1': [-0.04, 0.0, 0.0228],
                    'Body': [0, 0, 0]}


class HapticsForwardModel():
    """
    Haptics forward model for AoMR Shape grammar
    Creates 3D scene according to given shape representation 
    and uses VTK to render 3D scene to VRML file and GraspIt
    to calculate joint angles when grasping the object
    """
    parts = ['Body', 'Bottom0', 'Bottom1', 'Front0', 'Front1', 'Top0', 'Top1', 'Ear0', 'Ear1']
    models_folder = './models/'
    graspit_tcp_ip = '0'
    graspit_tcp_port = 4765
    graspit_objects_folder = "/home/goker/Programs/Graspit/models/objects/"
    graspit_joint_count = 16
    graspit_grasp_count = 24 # number of grasps that aomrGetJointAngles returns
    vrml_filename = "obj.wrl"
    # 3D models of object parts are too small for GraspIt
    # we need to scale objects for them to be suitable for
    # grasping
    scale_factor = 800
    # required for viewing object (for testing)
    window_size = (200, 200)
    camera_pos = (.18, 0.0, 0.0) 
    camera_up = (0, 0, 1)
    def __init__(self, body_fixed=True):
        self.body_fixed = body_fixed
        # vtk objects for creating 3D scene
        self.vtkrenderer = vtk.vtkRenderer()
        self.vtkrenderer.SetBackground(1, 1, 1)
        self.vtkrender_window = vtk.vtkRenderWindow()
        self.vtkrender_window.AddRenderer(self.vtkrenderer)
        self.vtkrender_window.SetSize(self.window_size)
        self.vtkrender_window_interactor = vtk.vtkRenderWindowInteractor()
        self.vtkrender_window_interactor.SetRenderWindow(self.vtkrender_window)
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition([p*self.scale_factor for p in self.camera_pos]);
        self.camera.SetFocalPoint(0, 0, 0);
        self.camera.SetViewUp(self.camera_up);
        self.vtkrenderer.SetActiveCamera(self.camera);
        
        self.vtkvrml_exporter = vtk.vtkVRMLExporter()
        # vtk objects for reading, and rendering object parts
        self.vtkreader = {}
        self.vtkpolydata = {}
        self.vtkmapper = {}
        self.vtktransform = {}
        self.vtkfilter = {}
        # read each part from its stl file
        for part in self.parts:
            self.vtkreader[part] = vtk.vtkSTLReader()
            self.vtkreader[part].SetFileName(self.models_folder + part + '.stl')
            self.vtktransform[part] = vtk.vtkTransform()
            self.vtktransform[part].Scale(self.scale_factor, self.scale_factor, self.scale_factor)
            self.vtkfilter[part] = vtk.vtkTransformFilter()
            self.vtkfilter[part].SetInput(self.vtkreader[part].GetOutput())
            self.vtkfilter[part].SetTransform(self.vtktransform[part])
            self.vtkpolydata[part] = self.vtkfilter[part].GetOutput()
            self.vtkmapper[part] = vtk.vtkPolyDataMapper()
            self.vtkmapper[part].SetInput(self.vtkpolydata[part])
        if self.body_fixed:
            # actor for body part (every object has a part named body at origin)
            self.vtkbodyactor = vtk.vtkActor()
            self.vtkbodyactor.SetMapper(self.vtkmapper['Body'])
            self.vtkbodyactor.SetPosition(0, 0, 0)
            
        self.graspit_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.graspit_socket.connect((self.graspit_tcp_ip, self.graspit_tcp_port))
        
    
    def __convert_aomr_shape_state_to_parts(self, aomr_shape_state):
        parts = []
        positions = []
        for node, pos in aomr_shape_state.positions.iteritems():
            parts.append(aomr_shape_state.tree[node].tag.symbol)
            positions.append(pos)
        return parts, positions
    
    def render(self, *args):
        """
        Place object parts given in `parts` to locations given
        in `positions` and save object to VRML file and get joint
        angles from GraspIt
        Returns numpy array with size number of joint angles (16)
        """
        if len(args) == 1: # called with AoMRShapeState object
            parts, positions = args[0].convert_to_parts_positions()
        elif len(args) == 2:
            parts, positions = args[0], args[1]
        else:
            raise TypeError("Must be called with 1 or 2 parameters")
        
        self.__build_scene(parts, positions)
        self.vtkrender_window.Render()
        
        # export object to vrml file
        self.vtkvrml_exporter.SetInput(self.vtkrender_window)
        self.vtkvrml_exporter.SetFileName(self.graspit_objects_folder + self.vrml_filename)
        self.vtkvrml_exporter.Write()
        
        # get joint angles from graspit
        data = self._graspit_get_joint_angles()
        return data
    
    def _graspit_get_joint_angles(self):
        """
        Commands GraspIt to load object, autoGrasp it and gets
        joint angles through TCP socket
        Returns a numpy array of size # of joint angles x # of grasps
        """
        # call aomrGetJointAngles, this command loads the object and
        # returns joint angles for 24 grasps for each of which the object
        # is rotated 45 deg around x, y, z axis
        data = self.__graspit_send_command('aomrGetJointAngles\n')
        if data[0:4] == 'FAIL':
            raise Exception('GraspIt returned error. Cannot get joint angles.')
        
        # convert data to numpy array
        fdata = [float(s) for s in data.split('\n')]
        # check data is correct, it should contain number of joint angles + 2 elements
        if len(fdata) != self.graspit_joint_count * self.graspit_grasp_count:
            raise Exception('Problem with data returned from GraspIt. Unexpected size.')
        
        joint_angles = np.asarray(fdata)
        
        return joint_angles
    
    def __graspit_send_command(self, command):
        """
        Sends command to GraspIt over socket and 
        returns the response
        """
        self.graspit_socket.send(command)
        data = self.graspit_socket.recv(4096)
        return data.rstrip()
    
    def __build_scene(self, parts, positions):
        """
        Places parts to positions and adds them to scene
        Returns vtkRenderer
        """
        # clear scene
        self.vtkrenderer.RemoveAllViewProps()
        if self.body_fixed:
            # add body
            self.vtkrenderer.AddActor(self.vtkbodyactor)
        for part, position in zip(parts, positions):
            actor = vtk.vtkActor()
            actor.SetMapper(self.vtkmapper[part])
            # we scaled each object part, so we need to adjust positions accordingly
            actor.SetPosition([p*self.scale_factor for p in position]) 
            self.vtkrenderer.AddActor(actor)
                
    def _view(self, *args):
        """
        Views object in window
        Used for development and testing purposes
        """
        if len(args) == 1: # called with AoMRShapeState object
            parts, positions = args[0].convert_to_parts_positions()
        elif len(args) == 2:
            parts, positions = args[0], args[1]
        else:
            raise TypeError("Must be called with 1 or 2 parameters")
        
        self.__build_scene(parts, positions)
        self.vtkrender_window.Render()
        self.vtkrender_window_interactor.Start()
        
if __name__ == '__main__':
    # ---------------------------------------------------------
    # show image and view 3D model for an object
    parts = ['Top0', 'Bottom0', 'Ear0', 'Front0']
    positions = [actual_positions['Top0'], actual_positions['Bottom0'],
               actual_positions['Ear0'], actual_positions['Front0']]
    forward_model = HapticsForwardModel()
     
    forward_model._view(parts, positions)
    np.set_printoptions(precision=3, linewidth=200)
    render = forward_model.render(parts, positions)
    print np.reshape(render, (24,16))
    #np.save('1.npy', render)
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
#     forward_model = HapticsForwardModel()
#     
#     positions = []
#     for i, object in enumerate(objects):
#         del positions[:]
#         for part in object:
#             positions.append(actual_positions[part])
#             
#         render = forward_model.render(object, positions)
#         print render
#         np.save(repr(i+1) + '.npy', render)

     # ---------------------------------------------------------