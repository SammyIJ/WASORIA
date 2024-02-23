# main.py
import argparse
from operations.svd import load_mnist as load_mnist_svd, apply_svd, reconstruct_images, plot_images
from operations.eigen_VV import load_mnist as load_mnist_pca, compute_PCA, plot_eigenfaces
from operations.linear_regression import load_and_preprocess_mnist, compute_pca_and_predict_intensity, plot_predictions

def run_svd():
    images, labels = load_mnist_svd()
    U_k, S_k, Vt_k = apply_svd(images)
    reconstructed_images = reconstruct_images(U_k, S_k, Vt_k)
    plot_images(images, reconstructed_images)

def run_pca():
    images, _ = load_mnist_pca()  # Updated to use the new loading function
    eigenvalues, eigenvectors = compute_PCA(images)
    print("Eigenvalues:", eigenvalues[:10])  # Print the first 10 eigenvalues
    plot_eigenfaces(eigenvectors, k=9)  # Plot the first 9 eigenfaces

def run_pis():
    images_flat, _ = load_and_preprocess_mnist()  # Using the preprocessing function that flattens images
    _, y_test, predicted_sums = compute_pca_and_predict_intensity(images_flat)
    plot_predictions(y_test, predicted_sums)  # Plot actual vs predicted pixel intensity sums

def main():
    parser = argparse.ArgumentParser(description="Perform SVD, PCA, or Pixel Intensity Sum prediction on the MNIST dataset.")
    parser.add_argument("--run", type=str, choices=["svd", "pca", "pis"], help="Specify operation to perform: svd, pca, or pis for Pixel Intensity Sum prediction")
    
    args = parser.parse_args()

    if args.run == "svd":
        run_svd()
    elif args.run == "pca":
        run_pca()
    elif args.run == "pis":  
        run_pis()

if __name__ == "__main__":
    main()
