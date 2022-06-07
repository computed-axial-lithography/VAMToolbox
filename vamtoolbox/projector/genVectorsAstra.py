import numpy as np

def genVectorsAstra(angles,incline_angle,cone_angle = 0,distance_origin_source = 0):
    
    vectors = np.zeros((np.size(angles),12))
    angles = np.deg2rad(angles)
    incline_angle = np.deg2rad(incline_angle)


    if cone_angle == 0:
        for i,angle in np.ndenumerate(angles):
            # ray direction
            vectors[i,0] = np.sin(angle)
            vectors[i,1] = -np.cos(angle)
            vectors[i,2] = np.sin(incline_angle)

            # center of detector
            vectors[i,3] = 0
            vectors[i,4] = 0
            vectors[i,5] = 0

            # vector from detector pixel (0,0) to (0,1)
            vectors[i,6] = np.cos(angle)
            vectors[i,7] = np.sin(angle)
            vectors[i,8] = 0

            # vector from detector pixel (0,0) to (1,0)
            vectors[i,9] = np.sin(angle) * np.cos(incline_angle)
            vectors[i,10] = -np.cos(angle) * np.cos(incline_angle)
            vectors[i,11] = np.cos(incline_angle)
        
    else:

        for i,angle in np.ndenumerate(angles):
            # source
            vectors[i,0] = np.sin(angle) * distance_origin_source
            vectors[i,1] = -np.cos(angle) * distance_origin_source
            vectors[i,2] = np.sin(incline_angle) * distance_origin_source

            # center of detector
            vectors[i,3] = -np.sin(angle) * distance_origin_source
            vectors[i,4] = np.cos(angle) * distance_origin_source
            vectors[i,5] = -np.sin(incline_angle) * distance_origin_source

            # vector from detector pixel (0,0) to (0,1)
            vectors[i,6] = np.cos(angle)
            vectors[i,7] = np.sin(angle)
            vectors[i,8] = 0

            # vector from detector pixel (0,0) to (1,0)
            vectors[i,9] = np.sin(angle) * np.cos(incline_angle)
            vectors[i,10] = -np.cos(angle) * np.cos(incline_angle)
            vectors[i,11] = np.cos(incline_angle)

    return vectors