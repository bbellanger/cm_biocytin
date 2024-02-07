# Documentation Resources: https://imagej.net/plugins/snt/scripting
# Latest SNT API: https://javadoc.scijava.org/SNT/

# Let the graphs open for some time
import time
# stuff to export data
import pandas as pd
import itertools
# FIJI, scyjava and SNT
import imagej
from scyjava import *

ij = imagej.init(["sc.fiji:fiji", "org.morphonets:SNT"], mode="interactive")
ij.ui().showUI()

# Get the active SNTService instance
snt = ij.getContext().getService("sc.fiji.snt.SNTService")

# Import tools for reconstruction
ImagePlus = jimport('ij.ImagePlus')
LutLoader = jimport('ij.plugin.LutLoader')
Dataset = jimport('net.imagej.Dataset')
ImageJFunctions = jimport('net.imglib2.img.display.imagej.ImageJFunctions')
BitType = jimport('net.imglib2.type.logic.BitType')
FloatType = jimport('net.imglib2.type.numeric.real.FloatType')
UnsignedByteType = jimport('net.imglib2.type.numeric.integer.UnsignedByteType')
Tree = jimport('sc.fiji.snt.Tree')
FillConverter = jimport('sc.fiji.snt.FillConverter')
FillerThread = jimport('sc.fiji.snt.tracing.FillerThread')
Reciprocal = jimport('sc.fiji.snt.tracing.cost.Reciprocal')
ImgUtils = jimport('sc.fiji.snt.util.ImgUtils')
WindowManager = jimport('ij.WindowManager')

# Import tools for analysis
TreeAnalyzer = jimport('sc.fiji.snt.analysis.TreeAnalyzer')

# Modules for clustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
# SNT clustering
PersistenceAnalyzer = jimport('sc.fiji.snt.analysis.PersistenceAnalyzer')
AllenUtils = jimport('sc.fiji.snt.annotation.AllenUtils')
Viewer3D = jimport('sc.fiji.snt.viewer.Viewer3D')
ColorTables = jimport('net.imagej.display.ColorTables')


def copyAxes(dataset, out_dataset):
    # Copy scale and axis metadata to the output.
    # There's probably a better way to do this...
    for d in range(dataset.numDimensions()):
        out_dataset.setAxis(dataset.axis(d), d)


def showGrayMask(dataset, converter):
    # Create an image of the same type and dimension as the input dataset
    output = ij.op().run("create.img", dataset)
    # Map pixel values at fill node positions between the input and output
    # The input and output need not have the same dimensions, but must be the same type.
    # The only other requirement is that both RandomAccessibles are defined at the positions
    # of the fill nodes
    converter.convert(dataset, output)
    # Convert output Img to another Dataset
    output = ij.dataset().create(output)
    # Copy the scale and axis metadata to the output
    copyAxes(dataset, output)
    ij.ui().show(output)


def showBinaryMask(dataset, converter):
    # The memory efficient BitType is great for a binary mask
    output = ij.op().run("create.img", dataset, BitType())
    # convertBinary only expects the output RandomAccessible, since it is
    # just setting 1 at fill node positions
    converter.convertBinary(output)
    # Convert output Img to another Dataset
    output = ij.dataset().create(output)
    # Copy the scale and axis metadata to the output
    copyAxes(dataset, output)
    ij.ui().show(output)

def showDistanceMap(dataset, converter, filename):
    # The node distance is stored internally as a Double,
    # but we can convert it to Float to display it
    output = ij.op().run("create.img", dataset, FloatType())
    converter.convertDistance(output)
    # Convert output Img to another Dataset
    output = ij.dataset().create(output)
    # Copy the scale and axis metadata to the output
    copyAxes(dataset, output)
    # Convert to ImagePlus so we can add a calibration bar overlay
    distanceImp = ij.convert().convert(output, ImagePlus)
    distanceImp.getProcessor().setColorModel(LutLoader.getLut("fire"))
    # The maximum pixel value is small, likely less than 1.
    # Reset the display range min-max so we can see this narrow band of intensities
    distanceImp.resetDisplayRange()
    # Add a calibration bar to visualize the distance measure
    ij.IJ.run(
        distanceImp,
        "Calibration Bar...",
        "location=[Upper Right] fill=White label=Black number=20 decimal=4 font=8 zoom=1.2 overlay"
    )
    distanceImp.setTitle("Annotated Distance Map")
    distanceImp.show()
    # Save the image in output/img
    ij.IJ.save(distanceImp, "../output/img/" + f'{filename}_DistanceMap.tif')

def showLabelMask(dataset, converter):
    # Choose the integer type based on the cardinality of
    # the input list of FillerThreads. If there are less than
    # 256, choose UnsignedByteType. If there are more than 255 but less
    # than 65536, choose UnsignedShortType, etc.
    # Assigned labels start at 1. 0 is reserved for background.
    output = ij.op().run("create.img", dataset, UnsignedByteType())
    converter.convertLabels(output)
    # Convert output Img to another Dataset
    output = ij.dataset().create(output)
    # Copy the scale and axis metadata to the output
    copyAxes(dataset, output)
    # I'm sure there is an ImageJ2 way of doing this...
    labelImp = ij.convert().convert(output, ImagePlus)
    labelImp.getProcessor().setColorModel(LutLoader.getLut("glasbey_on_dark"))
    labelImp.setTitle("Label Mask")
    labelImp.show()

def tracer(filename):
    # Paths for traces and file
    filepath = f'../input/{filename}.tif'
    trace_path = f'../input/{filename}.traces'
    # Load personal traces and stack for reconstruction
    tree = Tree(trace_path)
    dataset = ij.io().open(filepath)
    # Assign the scale metadata of the image to the Tree object
    # Otherwise, it won't know the mapping from world coordinates to voxel coordinates
    tree.assignImage(dataset)
    # Compute the minimum and maximum pixel intensities in the image.
    # These values are used by the search cost function to rescale
    # image values to a standardized interval (0.0 - 255.0) prior to
    # taking the reciprocal, which is used as the cost
    # of moving to that neighboring voxel. This cost is post-multiplied by physical distance.
    # Therefore, it is important to make sure the image is spatially calibrated before processing.
    # You could also try out different costs in the sc.fiji.snt.tracing.cost package.
    # One thing to note is that different costs will require *very* different thresholds, as the
    # distance magnitudes depend on the underlying function of intensity. For example,
    # the costs for OneMinusErf can go as low as 1.0 x 10^-75, so selecting a good threshold
    # is usually easier done interactivly via the GUI.
    min_max = ij.op().stats().minMax(dataset)
    cost = Reciprocal(min_max.getA().getRealDouble(), min_max.getB().getRealDouble())
    # Build the filler instance with a manually selected threshold
    threshold = 0.02
    fillers = []
    for path in tree.list():
        filler = FillerThread(dataset, threshold, cost)
        # Every point in the Path will be a seed point for the Dijkstra search.
        # Alternatively, you could construct a single FillerThread with all the Paths,
        # or for subsets of Paths, etc
        # setSourcePaths() expects a list, so wrap the Path in one
        filler.setSourcePaths([path])
        # Make sure this is set to True, otherwise the process will run until it has explored
        # every voxel in the input image. This can be slow and eat up memory, especially if the image is large.
        filler.setStopAtThreshold(True)
        # Set this to false to save memory,
        # the extra nodes are only relevant if you want to
        # resume progress using the same FillerThread instance,
        # which in most cases is unneccessary since the process is relatively quick.
        filler.setStoreExtraNodes(False)
        # Now run it. This could also be done in a worker thread
        filler.run()
        fillers.append(filler)
    # FillConverter accepts a list of (completed) FillerThread instances
    converter = FillConverter(fillers)
    showBinaryMask(dataset, converter)
    showGrayMask(dataset, converter)
    showLabelMask(dataset, converter)
    # For some reason, the distance map
    # must be shown last for the scale bar to show correctly??
    showDistanceMap(dataset, converter, filename)
    time.sleep(50)


# Clustering Morphologies Using Persitent Homology
# Output a dendrogram of the population from the batch
# def fetch_reconstruction(id_list):
#     tree_list: []

# Extract csv file with all the features from the reconstruction file
def extractcsv(filename, data):
    # Load personal traces and stack for TreeAnalyzer
    tree = Tree(f'../input/{filename}.traces')
    dataset = ij.io().open(f'../input/{filename}.tf')
    # Instantiate the analyzer
    analyzer = TreeAnalyzer(tree)
    # print("Highest Path order: ", analyzer.getHighestPathOrder())
    # Extract the same variable as for the cm_golgi staining project
    # and use the same exact name for the variables
    Sum_Length = analyzer.getCableLength()
    # Sum_N_tips = analyzer.getNtips()
    Avg_Branch_pathlength = analyzer.getAvgBranchLength()
    # Nb_branchpoint = analyzer.getNbranchPoints()
    Avg_Width = analyzer.getWidth()
    # Avg_Height =
    Avg_Depth = analyzer.getDepth()
    Avg_Partition_asymmetry = analyzer.getAvgPartitionAsymmetry()
    # Avg_Bif_ampl_local =
    # Avg_Bif_ampl_remote =
    # Avg_Bif_tilt_local =
    # Avg_Bif_tilt_remote =
    # Avg_Bif_torque_local =
    # Avg_Bif_torque_remote =
    Avg_Contraction = analyzer.getAvgContraction()
    Avg_Fractal_Dim = analyzer.getAvgFractalDimension()
    # Avg_Terminal_degree =
    # Max_Bif_ampl_local =
    # Max_Bif_ampl_remote =
    Avg_Bif_ampl_remote = analyzer.getAvgRemoteBifAngle()
    # Max_Bif_tilt_local =
    # Max_Bif_tilt_remote =
    # Max_Bif_torque_local =
    # Max_Bif_torque_remote =
    # Max_Fractal_Dim =
    # Max_Branch_pathlength =
    # Max_Branch_Order =
    # Max_PathDistance =
    # Max_Helix =

    # Print each variable output in the terminal
    print(" ")
    print('---------------------------------')
    print(" ")
    print("Neuron: ", filename)
    print(" ")
    print("Cable length: ", Sum_Length)
    # print("Nb of tips: ", Sum_N_tips)
    # print("N Branch points: ", Nb_branchpoint)
    print("Highest Path order: ", analyzer.getHighestPathOrder())
    print("Avg branch length: ", Avg_Branch_pathlength)
    print("Avg depth", Avg_Depth)
    print("Avg Partition asymetry: ", Avg_Partition_asymmetry)
    print("Avg contraction: ", Avg_Contraction)
    print("Avg fractal dimension: ", Avg_Fractal_Dim)
    print("Avg remote bifurcation angle: ", Avg_Bif_ampl_remote)
    print("Avg width: ", Avg_Width)
    print(" ")
    print('---------------------------------')
    print(" ")

    # Build a dictionary from the extracted variables
    # Save a csv file from it with Pandas
    my_dictionary = {
        'image': filename,
        'Sum_Length': Sum_Length, 'Avg_Branch_pathlength': Avg_Branch_pathlength,
        'Avg_Width': Avg_Width, 'Avg_Depth': Avg_Depth, 'Avg_Partition_asymmetry': Avg_Partition_asymmetry,
        'Avg_Contraction': Avg_Contraction, 'Avg_Fractal_Dim': Avg_Fractal_Dim,
        'Avg_Bif_ampl_remote': Avg_Bif_ampl_remote
    }
    df_dictionary = pd.DataFrame.from_dict(my_dictionary, orient='index').T
    print(data)
    return df_dictionary
    time.sleep(25)
    # Instantiate a new sholl analyzer and process the same tree
    # shollanalyzer = ShollAnalyzer(tree)
    # print("Metrics: ", shollanalyzer.getMetrics())

def main():
    list_files = pd.read_fwf("../table/list.txt", header=0)
    data = []
    for img_id in list_files:
        filename = img_id
        print(' ')
        print('Input list:')
        print(list_files)
        # print(img_id)
        # print(filename)
        tracer(filename)
        dfdict = extractcsv(filename, data)
        data.append(dfdict)
    data = pd.concat(data, ignore_index=True)
    data.to_csv(f'../build/data.csv', index=False)
    print(f'Saved:')
    print(f'{list_files}')
    print('features in build directory')


# def clustering_dendrogram():
#     list_files = pd.read_fwf("../table/list.txt", header=0)
#     tree_list = []
#     for filename in list_files:
#         # Path for traces and files
#         file_path = f'../input/{filename}.tif'
#         trace_path = (f'../input/{filename}.traces')
#         # Load personnal traces and stack
#         tree = Tree(trace_path)
#         dataset = ij.io().open(file_path)
#         # Assign the scale metadata to the tree object
#         # Transmits world coordinates too voxel coordinates
#         tree.assignImage(dataset)
#         # tree_map[tree.getLabel()[0:6]] = tree
#         # tree.setLabel(tree.getLabel() + ' ' + group_label)
#         tree_list.append(tree)
#         return tree_list


# main()
# clustering_dendrogram()