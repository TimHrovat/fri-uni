import numpy as np
from PIL import Image
import json

# Load test data
with open('primeri/1.json', 'r') as f:
    data = json.load(f)

# Load image
slika_file = Image.open('primeri/' + data['slika']).convert('L')
slika = np.array(slika_file, dtype=np.int16)

print(f"Image shape: {slika.shape}")
print(f"Image dtype: {slika.dtype}")
print(f"Image range: [{slika.min()}, {slika.max()}]")
print(f"Threshold: {data['prag']}")
print(f"Expected MI: {data['MI']}")

# Test the function with debug output
try:
    from naloga4 import naloga4
    print("Function imported successfully")

    # Try calling the function
    print("Calling naloga4...")
    result = naloga4(slika, data['prag'])
    print(f"Result: {result}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()