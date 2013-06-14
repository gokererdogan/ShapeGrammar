# -*- coding: utf-8 -*-
'''
Created on May 9, 2013

@author: gerdogan

VTK test code
'''

# 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

reader = vtk.vtkSTLReader()
reader2 = vtk.vtkSTLReader()
reader3 = vtk.vtkSTLReader()
reader4 = vtk.vtkSTLReader()
reader5 = vtk.vtkSTLReader()

model_folder = './models/'
filename = model_folder + "Body.stl"
reader.SetFileName(filename)
polyDataOutput = reader.GetOutput()

filename = model_folder + "Front0.stl"
reader2.SetFileName(filename)
polyDataOutput2 = reader2.GetOutput()

filename = model_folder + "Bottom1.stl"
reader3.SetFileName(filename)
polyDataOutput3 = reader3.GetOutput()

filename = model_folder + "Top1.stl"
reader4.SetFileName(filename)
polyDataOutput4 = reader4.GetOutput()

filename = model_folder + "Ear1.stl"
reader5.SetFileName(filename)
polyDataOutput5 = reader5.GetOutput()

scale_tr = vtk.vtkTransform()
scale_tr.Scale(5,5,5)
scale_filter = vtk.vtkTransformFilter()
scale_filter.SetInput(polyDataOutput5)
scale_filter.SetTransform(scale_tr)
polyDataOutput5_scaled = scale_filter.GetOutput()

# Create a mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInput(polyDataOutput)
body = vtk.vtkActor()
body.SetMapper(mapper)

mapper2 = vtk.vtkPolyDataMapper()
mapper2.SetInput(polyDataOutput2)
front = vtk.vtkActor()
front.SetMapper(mapper2)
front.SetPosition(.0535, 0, .0054)

mapper3 = vtk.vtkPolyDataMapper()
mapper3.SetInput(polyDataOutput3)
bottom = vtk.vtkActor()
bottom.SetMapper(mapper3)
bottom.SetPosition(-.0025, 0, -.0228)

mapper4 = vtk.vtkPolyDataMapper()
mapper4.SetInput(polyDataOutput4)
top = vtk.vtkActor()
top.SetMapper(mapper4)
top.SetPosition(-.0191, 0, .0259)

mapper5 = vtk.vtkPolyDataMapper()
mapper5.SetInput(polyDataOutput5_scaled)
ear = vtk.vtkActor()
ear.SetMapper(mapper5)
ear.SetPosition(-.0237, 0, .0341)

camera = vtk.vtkCamera();
camera.SetPosition(.16, -.16, .16);
camera.SetFocalPoint(0, 0, 0);
camera.SetViewUp(0, 0, 1)

# x, y, z lines
# create source
xl = vtk.vtkLineSource()
xl.SetPoint1(-10,0,0)
xl.SetPoint2(10,0,0)
yl = vtk.vtkLineSource()
yl.SetPoint1(0,-10,0)
yl.SetPoint2(0,10,0)
zl = vtk.vtkLineSource()
zl.SetPoint1(0,0,-10)
zl.SetPoint2(0,0,10)
 
# mapper
mapperx = vtk.vtkPolyDataMapper()
mapperx.SetInput(xl.GetOutput())
mappery = vtk.vtkPolyDataMapper()
mappery.SetInput(yl.GetOutput())
mapperz = vtk.vtkPolyDataMapper()
mapperz.SetInput(zl.GetOutput())

# actor
actorx = vtk.vtkActor()
actorx.SetMapper(mapperx)
actory = vtk.vtkActor()
actory.SetMapper(mappery)
actorz = vtk.vtkActor()
actorz.SetMapper(mapperz)

# color actor
actorx.GetProperty().SetColor(1,0,0)
actory.GetProperty().SetColor(0,1,0)
actorz.GetProperty().SetColor(0,1,1)

# Visualize
renderer = vtk.vtkRenderer()
renderer.SetActiveCamera(camera);
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(600, 600)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
 
renderer.AddActor(body)
renderer.AddActor(front)
renderer.AddActor(bottom)
renderer.AddActor(top)
renderer.AddActor(ear)
renderer.AddActor(actorx)
renderer.AddActor(actory)
renderer.AddActor(actorz)

#renderer.RemoveAllViewProps()

renderer.SetBackground(0, 0, 0) # Background color
renderWindow.Render()
 
vrml = vtk.vtkVRMLExporter()
vrml.SetInput(renderWindow)
vrml.SetFileName("/home/goker/Graspit/models/objects/test.wrl")
vrml.Write()
renderWindowInteractor.Start()

#vtk_win_im = vtk.vtkWindowToImageFilter()
#vtk_win_im.SetInput(renderWindow)
#vtk_win_im.Update()
#
##writer = vtk.vtkPNGWriter()
##writer.SetFileName("screenshot.png")
##writer.SetInput(vtk_win_im.GetOutput())
##writer.Write()
#
#vtk_image = vtk_win_im.GetOutput()
#
#height, width, _ = vtk_image.GetDimensions()
#vtk_array = vtk_image.GetPointData().GetScalars()
#components = vtk_array.GetNumberOfComponents()
#
#arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
#print arr