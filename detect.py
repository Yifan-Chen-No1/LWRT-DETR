import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/DOTA-PyCharm-3/weights/best.pt') # select your model.pt path
    model.predict(source='C:/Users/NINGMEI/Desktop/影响',
                  project='runs/detect',
                  name='影响',
                  conf=0.2,
                  iou=0.8,
                  save=True,
                  line_width=2,
                  show_conf=False,
                  show_labels=False,
                  visualize=True # visualize model features maps
                  )