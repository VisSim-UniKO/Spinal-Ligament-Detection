import os
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
import vtk
import ALPACA
import scipy.spatial as spatial
import time
from pathlib import Path
import ExtractCenterline
from vtk.util.numpy_support import vtk_to_numpy

import SpineLib
import vtk_convenience as conv


class PointSearch(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Ligament Point Search"
        self.parent.categories = ["VisSimTools"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ivanna Kramer, Lara Blomenkamp (VisSim Research Group)"]
        self.parent.helpText =''''''
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """University of Koblenz"""

#
# PointSearchWidget
#

class PointSearchWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):

        ScriptedLoadableModuleWidget.setup(self)

        scriptPath = os.path.dirname(os.path.abspath(__file__))

        collapsibleButton = ctk.ctkCollapsibleButton()
        collapsibleButton.text = "Point Search"
        self.layout.addWidget(collapsibleButton)
        formLayout = qt.QFormLayout(collapsibleButton)

        # source model selector
        self.sourceModelSelector = slicer.qMRMLNodeComboBox()
        self.sourceModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.sourceModelSelector.selectNodeUponCreation = True
        self.sourceModelSelector.addEnabled = False
        self.sourceModelSelector.removeEnabled = False
        self.sourceModelSelector.noneEnabled = False
        self.sourceModelSelector.showHidden = False
        self.sourceModelSelector.showChildNodeTypes = False
        self.sourceModelSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceModelSelector.setToolTip("Source model:")
        formLayout.addRow("Source Model: ", self.sourceModelSelector)

        # source landmarks selector
        self.sourceLandmarksSelector = slicer.qMRMLNodeComboBox()
        self.sourceLandmarksSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.sourceLandmarksSelector.selectNodeUponCreation = True
        self.sourceLandmarksSelector.addEnabled = False
        self.sourceLandmarksSelector.removeEnabled = False
        self.sourceLandmarksSelector.noneEnabled = False
        self.sourceLandmarksSelector.showHidden = False
        self.sourceLandmarksSelector.showChildNodeTypes = False
        self.sourceLandmarksSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceLandmarksSelector.setToolTip("Source landmarks:")
        formLayout.addRow("Source Landmarks: ", self.sourceLandmarksSelector)

        # target landmarks selector
        self.targetLandmarksSelector = slicer.qMRMLNodeComboBox()
        self.targetLandmarksSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetLandmarksSelector.selectNodeUponCreation = True
        self.targetLandmarksSelector.addEnabled = False
        self.targetLandmarksSelector.removeEnabled = False
        self.targetLandmarksSelector.noneEnabled = False
        self.targetLandmarksSelector.showHidden = False
        self.targetLandmarksSelector.showChildNodeTypes = False
        self.targetLandmarksSelector.setMRMLScene(slicer.mrmlScene)
        self.targetLandmarksSelector.setToolTip("Target landmarks:")
        formLayout.addRow("Target Landmarks: ", self.targetLandmarksSelector)




        
        # Process Button
        self.process = qt.QPushButton("Run Point Search Process")
        self.process.enabled = True
        self.process.setStyleSheet("font: bold; background-color: blue; font-size: 12px; height: 48px; width: 120px;")
        formLayout.addRow(self.process)
        self.process.connect('clicked(bool)', self.onProcessButton)

        
        ###################################################################################

        # Add vertical spacer
        self.layout.addStretch(1)
        
        # enddef setup


    def cleanup(self):
        pass

    def onProcessButton(self):
        sourceModel = self.sourceModelSelector.currentNode()
        sourceLandmarks = self.sourceLandmarksSelector.currentNode()
        targetLandmarks = self.targetLandmarksSelector.currentNode()
        logic = PointSearchLogic()
        logic.run()


#
# PointSearchLogic
#

class PointSearchLogic(ScriptedLoadableModuleLogic):


    def find_polydata_neighbors(self, polydata, point, radius):
        normals = np.array(list(conv.iter_normals(polydata)))
        vertices = np.array(list(conv.iter_points(polydata)))
        point_tree = spatial.cKDTree(vertices)
        neighbors = point_tree.query_ball_point(point, radius)
        neighbor_vertices = vertices[neighbors]
        neighbor_normals = normals[neighbors]

        return neighbor_vertices, neighbor_normals
    
    def edge_model(self, model, radius):
        
        distances = []
        vertices = np.array(list(conv.iter_points(model.GetPolyData())))

        point_tree = spatial.cKDTree(vertices)

        for i, point in enumerate(vertices):
            neighbors = point_tree.query_ball_point(point, radius)
            neighbor_vertices = vertices[neighbors]

            # distance of centroid to query point
            centroid = np.mean(neighbor_vertices, axis=0)
            distance = np.linalg.norm(centroid - point)
            distances.append(distance)
        

        # Convert distances to a VTK array
        vtk_distances = vtk.vtkFloatArray()
        vtk_distances.SetNumberOfComponents(1)
        vtk_distances.SetName("Distances")

        for distance in distances:
            vtk_distances.InsertNextValue(distance)

        # Set the scalar values for vertices in the polydata
        model.GetPolyData().GetPointData().SetScalars(vtk_distances)

    
    def find_landmarks_body(self, polydata, sourceLandmarks, targetLandmarksNode, radius):

        vertices = np.array(list(conv.iter_points(polydata)))
        body_centroid = np.mean(sourceLandmarks, axis=0)

        point_tree = spatial.cKDTree(vertices)

        # enum landmarks
        for i, lm in enumerate(sourceLandmarks):
            try:
                # vector from centroid to landmark
                vector = lm - body_centroid
                vector = vector / np.linalg.norm(vector)

                up = np.array([0, 0, 1])
                plane_normal = np.cross(vector, up)

                intersection = conv.cut_plane(polydata, lm, plane_normal)

                # get intersection point with highest distance
                intersection_vertices = np.array(list(conv.iter_points(intersection)))
                intersection_tree = spatial.cKDTree(intersection_vertices)
                intersection_distances = []

                # find neighbors of landmark in intersection tree
                intersection_neighbors = intersection_tree.query_ball_point(lm, radius*2)
                query_points = [intersection_vertices[ne] for ne in intersection_neighbors]

                # find distance value of int_neighbor_vertices in model
                for q in query_points:
                    query_neighbors = point_tree.query_ball_point(q, radius)
                    query_neighbor_vertices = [vertices[qn] for qn in query_neighbors]

                    # distance of centroid to query point
                    centroid = np.mean(query_neighbor_vertices, axis=0)
                    distance = np.linalg.norm(centroid - q)
                    intersection_distances.append(distance)
                
                # find index of xlint_neighbor with highest distance
                max_index = np.argmax(intersection_distances)
                max_vertex = query_points[max_index]
                targetLandmarksNode.AddFiducialFromArray(max_vertex)

            except:
                continue

    def find_landmarks_proc(self, model, sourceLandmarks, targetLandmarksNode, radius, plane_normal):

        # enum landmarks
        for i, lm in enumerate(sourceLandmarks):
            try:
                intersection = conv.cut_plane(model.GetPolyData(), lm, plane_normal)
                intersection_vertices = np.array(list(conv.iter_points(intersection)))
                intersection_tree = spatial.cKDTree(intersection_vertices)

                # find closest point to landmark in intersection tree
                distance, closest_point_index = intersection_tree.query(lm)
                closest_point = intersection_vertices[closest_point_index]
                targetLandmarksNode.AddFiducialFromArray(closest_point)

            except:
                continue

    def find_landmarks_closest(self, model, sourceLandmarks, targetLandmarksNode, radius):

        # enum landmarks
        for i, lm in enumerate(sourceLandmarks):
            try:
                vertices = np.array(list(conv.iter_points(model.GetPolyData())))
                point_tree = spatial.cKDTree(vertices)

                # find closest point to landmark in intersection tree
                distance, closest_point_index = point_tree.query(lm)
                closest_point = vertices[closest_point_index]
                targetLandmarksNode.AddFiducialFromArray(closest_point)
            except:
                continue

    def find_landmarks_radius(self, model, sourceLandmarks, targetLandmarksNode, radius):

        vertices = np.array(list(conv.iter_points(model.GetPolyData())))
        point_tree = spatial.cKDTree(vertices)

        # enum landmarks
        for i, lm in enumerate(sourceLandmarks):
            try:
                # find closest point to landmark in model
                point_distances = []
                distance, closest_point_index = point_tree.query(lm)
                query_point = vertices[closest_point_index]

                # find neighbors of landmark in intersection tree
                neighbors = point_tree.query_ball_point(query_point, radius*2)
                neighbor_points = [vertices[ne] for ne in neighbors]

                # find distance value of int_neighbor_vertices in model
                for q in neighbor_points:
                    query_neighbors = point_tree.query_ball_point(q, radius)
                    query_neighbor_vertices = [vertices[qn] for qn in query_neighbors]

                    # distance of centroid to query point
                    centroid = np.mean(query_neighbor_vertices, axis=0)
                    distance = np.linalg.norm(centroid - q)
                    point_distances.append(distance)
                
                # find index of xlint_neighbor with highest distance
                max_index = np.argmax(point_distances)
                max_vertex = neighbor_points[max_index]
                targetLandmarksNode.AddFiducialFromArray(max_vertex)
            except:
                continue
            

                
    

    def alpaca(self, source, target):
        alpacaLogic = ALPACA.ALPACALogic()
        alpacaWidget = ALPACA.ALPACAWidget()
        (sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize, scaling) = alpacaLogic.runSubsample(source, target, False, alpacaWidget.parameterDictionary, False)
        transformMatrix, similarityFlag = alpacaLogic.estimateTransform(sourcePoints,targetPoints,sourceFeatures,targetFeatures,voxelSize,False,alpacaWidget.parameterDictionary)
        vtkTransformMatrix = alpacaLogic.itkToVTKTransform(transformMatrix, similarityFlag)
        registrationMatrix = vtkTransformMatrix.GetMatrix()
        transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
        transformNode.SetMatrixTransformToParent(registrationMatrix)
        source.SetAndObserveTransformNodeID(transformNode.GetID())

        return registrationMatrix
    

    

    def find_ligament_landmarks(self, sourceModel, sourceLandmarksNode):

        ##############################################################################################
        # ALPACA registration

        # registrationMatrix = self.alpaca(sourceModel, targetModel)

        ##############################################################################################
        
        # edge model
        #self.edge_model(sourceModel, radius=8)

        # add fiducial list
        targetLandmarksNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')

        # delete all points in targetLandmarksNode
        targetLandmarksNode.RemoveAllControlPoints()

        sourceLandmarks = slicer.util.arrayFromMarkupsControlPoints(sourceLandmarksNode)
        bodyLandmarks, lf_Landmarks, isl_Landmarks, itl_Landmarks, cl_Landmarks, ssl_Landmarks = [],[],[],[],[],[]

        for i in range(0, len(sourceLandmarks)):
            if "ALL" in sourceLandmarksNode.GetNthFiducialLabel(i) or "PLL" in sourceLandmarksNode.GetNthFiducialLabel(i):
                bodyLandmarks.append(sourceLandmarks[i])
            elif "LF_L" in sourceLandmarksNode.GetNthFiducialLabel(i) or "LF_R" in sourceLandmarksNode.GetNthFiducialLabel(i):
                lf_Landmarks.append(sourceLandmarks[i])
            elif "ISL" in sourceLandmarksNode.GetNthFiducialLabel(i):
                isl_Landmarks.append(sourceLandmarks[i])
            elif "ITL" in sourceLandmarksNode.GetNthFiducialLabel(i):
                itl_Landmarks.append(sourceLandmarks[i])
            elif "CL" in sourceLandmarksNode.GetNthFiducialLabel(i):
                cl_Landmarks.append(sourceLandmarks[i])
            elif "SSL" in sourceLandmarksNode.GetNthFiducialLabel(i):
                ssl_Landmarks.append(sourceLandmarks[i])

        body_poly = conv.clip_plane(sourceModel.GetPolyData(), conv.calc_center_of_mass(sourceModel.GetPolyData()), np.array([0, 1, 0]))
        
        self.find_landmarks_body(body_poly, bodyLandmarks, targetLandmarksNode, radius=5)
        self.find_landmarks_proc(sourceModel, isl_Landmarks, targetLandmarksNode, radius=5, plane_normal = np.array([0, 1, 0]))
        self.find_landmarks_proc(sourceModel, lf_Landmarks, targetLandmarksNode, radius=5, plane_normal = np.array([1, 0, 0]))
        self.find_landmarks_radius(sourceModel, itl_Landmarks, targetLandmarksNode, radius=20)
        self.find_landmarks_radius(sourceModel, ssl_Landmarks, targetLandmarksNode, radius=20)
        self.find_landmarks_closest(sourceModel, cl_Landmarks, targetLandmarksNode, radius=20)


    def performLandmarksSearch(self, vt):

        #compute center of mass 
        center = vtk.vtkCenterOfMass()
        center.SetInputData(vt.geometry)
        center.Update()
        print(f'center of mass: {center.GetCenter()}')

        postPolyData = conv.clip_plane(vt.geometry, center.GetCenter(), -vt.orientation.a)
        postVertNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        postVertNode.SetAndObservePolyData(postPolyData)
        otwm = vt.objectToWorldMatrix
        otwm.Invert()
        SpineLib.SlicerTools.transformVertebraObjects(otwm, [postVertNode])

        #compute center of mass for posterior part
        postCenter = vtk.vtkCenterOfMass()
        postCenter.SetInputData(postPolyData)
        postCenter.Update()
        print(f'center of mass in posterior part: {postCenter.GetCenter()}')
        
        leftPostPolyData = conv.clip_plane(postPolyData, postCenter.GetCenter(), -vt.orientation.r)
        postLeftVertNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        postLeftVertNode.SetAndObservePolyData(leftPostPolyData)
        rightPostPolyData = conv.clip_plane(postPolyData, postCenter.GetCenter(), vt.orientation.r)
        postRightVertNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        postRightVertNode.SetAndObservePolyData(rightPostPolyData)

        registrationLandmarks = list(vt.landmarks.__dict__.values())

        l_npVertPoints = slicer.util.arrayFromModelPoints(postLeftVertNode)
        r_npVertPoints = slicer.util.arrayFromModelPoints(postRightVertNode)

        SpineLib.SlicerTools.removeNodes([postLeftVertNode, postRightVertNode])
        
        print(l_npVertPoints[:, 0].argsort())

        left   = l_npVertPoints[l_npVertPoints[:, 0].argsort()[0]]
        l_post = l_npVertPoints[l_npVertPoints[:, 1].argsort()[0]]
        l_inf  = l_npVertPoints[l_npVertPoints[:, 2].argsort()[0]]
        l_sup  = l_npVertPoints[l_npVertPoints[:, 2].argsort()[-1]]
        right  = r_npVertPoints[r_npVertPoints[:, 0].argsort()[-1]]
        r_post = r_npVertPoints[r_npVertPoints[:, 1].argsort()[0]]
        r_inf  = r_npVertPoints[r_npVertPoints[:, 2].argsort()[0]]
        r_sup  = r_npVertPoints[r_npVertPoints[:, 2].argsort()[-1]]

        posteriorLandmarks = [left, l_post, l_inf, l_sup, right, r_post, r_inf, r_sup]
        # for lm in posteriorLandmarks:
        #     registrationLandmarks.append(lm)

        # create fiducial node for registration landmarks
        registrationLandmarksNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        registrationLandmarksNode.CreateDefaultDisplayNodes()
        registrationLandmarksNode.SetName("registrationLandmarks")
        for lm in posteriorLandmarks:
            registrationLandmarksNode.AddFiducialFromArray(lm)

        otwm.Invert()
        SpineLib.SlicerTools.transformVertebraObjects(otwm, [registrationLandmarksNode])

        for lm in registrationLandmarks:
            registrationLandmarksNode.AddFiducialFromArray(lm)
        
        return registrationLandmarksNode




    def run(self, sourceModels, sourceLandmarks, targetModels):

        ####################################################################################################################################################
        # SPINE INITIALIZATION

        # init source spine
        source_Geometries   =   [node.GetPolyData() for node in sourceModels]
        source_Spine        =   SpineLib.Spine(geometries=source_Geometries, max_angle=45.0)

        # init source ligament landmarks
        source_LandmarkNodes   = list(sourceLandmarks)

        # init target spine
        target_Geometries   =   [node.GetPolyData() for node in targetModels]
        target_Spine        =   SpineLib.Spine(geometries=target_Geometries, max_angle=45.0)


        ################################################################################################################################################

        # enumerate vertebrae
        for i, vt in enumerate(target_Spine.vertebrae):

            source_reg_landmarks_node = self.performLandmarksSearch(source_Spine.vertebrae[i])
            target_reg_landmarks_node = self.performLandmarksSearch(target_Spine.vertebrae[i])

            frw = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLFiducialRegistrationWizardNode')
            frw.SetRegistrationModeToSimilarity()
            transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
            frw.SetOutputTransformNodeId(transformNode.GetID())
            frw.SetAndObserveFromFiducialListNodeId(source_reg_landmarks_node.GetID())
            frw.SetAndObserveToFiducialListNodeId(target_reg_landmarks_node.GetID())

            source_LandmarkNodes[i].SetAndObserveTransformNodeID(transformNode.GetID())
            source_LandmarkNodes[i].HardenTransform()

            # LIGAMENT LANDMARK DETECTION
            self.find_ligament_landmarks(targetModels[i], sourceLandmarks[i])
            


class PointSearchTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_PointSearch1()

    def test_PointSearch1(self):
         
        self.delayDisplay("Starting the test")
        
        # load segmentation models
        scriptPath = os.path.dirname(os.path.abspath(__file__))
        # get parent directory


        verse_Directory = os.path.join(Path(scriptPath).parent, "Datasets/test/Verse005")
        verse_files = slicer.util.getFilesInDirectory(verse_Directory)
        verse_model_nodes = []

        for file in verse_files:
            if file.endswith(".stl") or file.endswith(".obj"):
                node = slicer.util.loadModel(file)
                verse_model_nodes.append(node)

        diers_Directory = os.path.join(Path(scriptPath).parent, "Datasets/test/dataset_diers")
        diers_files = slicer.util.getFilesInDirectory(diers_Directory)
        diers_model_nodes = []
        diers_landmark_nodes = []

        for file in diers_files:
            if file.endswith(".stl") or file.endswith(".obj"):
                node = slicer.util.loadModel(file)
                diers_model_nodes.append(node)
            if file.endswith(".fcsv"):
                node = slicer.util.loadMarkups(file)
                diers_landmark_nodes.append(node)

        logic = PointSearchLogic()
        logic.run(diers_model_nodes, diers_landmark_nodes, verse_model_nodes)
        


        # ###### RUN REGISTRATION PROCESS ############
        # scriptPath = os.path.dirname(os.path.abspath(__file__))
        # sawboneDir = os.path.join(scriptPath, "LumbarSpineSawbones")
        # logic  = PointSearchLogic()
        # logic.run(sawboneDir)
        # ############################################




        self.delayDisplay('Test passed!')