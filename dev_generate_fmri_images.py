# Script for generating images of See&Grasp dataset for FMRI experiment
# The images for our experiment are generated in vision_forward_model.py file.
#
# Goker Erdogan
# 21 March 2014


import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import scipy.misc

reader = vtk.vtkSTLReader()

objects = [5, 9, 14, 20, 21, 23, 31, 32]
current_obj = 8
model_folder = './models/SeeGrasp/'
object_id = objects[current_obj-1]
filename = model_folder + str(object_id) + ".stl"
reader.SetFileName(filename)
polyDataOutput = reader.GetOutput()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInput(polyDataOutput)
body = vtk.vtkActor()
body.SetMapper(mapper)

camera = vtk.vtkCamera();
camera.SetPosition(270, 160, 270);
camera.SetFocalPoint(0, 0, 0);
camera.SetViewUp(0, 1, 0)

renderer = vtk.vtkRenderer()
renderer.SetActiveCamera(camera);
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(600, 600)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

renderer.AddActor(body)

view_angle = 0
angle_step = 1
camera_distance = 270 * np.sqrt(2)
camera_y = 160

for i in range(360/angle_step):
    save_fname = '/home/goker/aomr_fmri/' + str(i+(360*(current_obj-1))+1) + '.png'
    view_angle += angle_step
    view_angle_radian = np.pi * (view_angle / 180.0)
    camera_x = camera_distance * np.sin(view_angle_radian)
    camera_z = camera_distance * np.cos(view_angle_radian)
    camera.SetPosition((camera_x, camera_y, camera_z))
    # save image
    renderWindow.Render()
    win_im = vtk.vtkWindowToImageFilter()
    win_im.SetInput(renderWindow)
    win_im.Update()
    vtk_image = win_im.GetOutput()
    height, width, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
    # convert RGB image to grayscale image 
    arr = np.sum(arr, 2) / 3.0
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    scipy.misc.imsave(save_fname, np.flipud(arr))
    

#renderWindow.Render()
#renderWindowInteractor.Start()

