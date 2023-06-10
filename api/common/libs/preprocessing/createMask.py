import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

if __name__ == "__main__":
    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl")
    ins.segmentImage("images/sample_images/sample2.jpg", show_bboxes=True, output_image_name="masked_image2.jpg")
