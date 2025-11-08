print("Starting minimal test...")

try:
    print("Importing numpy...")
    import numpy as np
    print("Numpy imported successfully")

    print("Importing naloga4...")
    from naloga4 import naloga4
    print("naloga4 imported successfully")

    print("Creating test data...")
    # Create a small test image
    test_image = np.random.rand(10, 10).astype(np.float64)
    test_threshold = 0.1

    print(f"Test image shape: {test_image.shape}")
    print(f"Test image range: [{test_image.min():.3f}, {test_image.max():.3f}]")

    print("Calling naloga4 with small test data...")
    result = naloga4(test_image, test_threshold)
    print(f"Result: {result}")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")