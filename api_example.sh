
curl -X POST http://localhost:5000/api/predict \
    -F "image=@testdata/car.jpg" \
    -F "task=REFERRING_EXPRESSION_SEGMENTATION" \
    -F "prompt=the car"