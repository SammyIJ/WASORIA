import cv2 as cv
from sam.image_preprocessing import canny_edge_detection, erosion, dilation

def main():
    image_paths = cv.imread('/home/sammy/Desktop/Samuel/dataset/image1.jpg')  
    edge_detected_images = canny_edge_detection(image_paths)

    # Apply erosion
    eroded_images = erosion(edge_detected_images, kernel_size=(5, 5), iterations=1)

    # Apply dilation
    dilated_images = dilation(edge_detected_images, kernel_size=(5, 5), iterations=1)

    # Output
    #output_images = dilated_images + eroded_images

  
    
    # You can save or display the processed images as needed
    # Example of saving the first eroded image:
    cv.imwrite('/home/wasoria-sam/PycharmProjects/Samuel/output', eroded_images[0])

    # Example of saving the first dilated image:
    cv.imwrite('/home/wasoria-sam/PycharmProjects/Samuel/output', dilated_images[0])

    # Example of saving the first output image:
    #cv.imwrite('/home/wasoria-sam/PycharmProjects/Samuel/output', output_images[0])

    if __name__ == "__main__":
        main()

