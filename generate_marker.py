import cv2
import numpy as np

def generate_aruco_marker(marker_id, marker_size, dictionary=cv2.aruco.DICT_6X6_250, save_path=None):
    """
    Generates an ArUco marker with the given ID and size.
    
    Parameters:
        marker_id (int): The ID of the marker to generate.
        marker_size (int): The size of the marker image in pixels.
        dictionary (cv2.aruco.Dictionary): The ArUco dictionary to use (default is 6x6_250).
        save_path (str, optional): Path to save the generated marker image. If None, it will not be saved.
    
    Returns:
        numpy.ndarray: The generated ArUco marker image.
    """
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    if save_path:
        cv2.imwrite(save_path, marker_img)
    
    return marker_img

if __name__ == "__main__":
    marker_id = 27  # Change this to the desired marker ID
    marker_size = 2000  # Change this to the desired marker size in pixels
    save_path = f"aruco_marker_{marker_id}.png"  # Path to save the marker image
    
    marker_img = generate_aruco_marker(marker_id, marker_size, save_path=save_path)
    
    # Display the generated marker
    cv2.imshow("ArUco Marker", marker_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
