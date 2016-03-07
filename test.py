import pyglet
import cv2
import cv
import os
from pytube import YouTube

def main():
  # source = pyglet.media.load('/home/ioana/sample/iorigins/media/_0nX-El-ySo_83_93.avi')
  source = pyglet.media.load('./bird.avi')
  frame1 = source.get_next_video_frame()
  frame2 = source.get_next_video_frame()
  frame3 = source.get_next_video_frame()
  video_id = 'mv89psg6zh4'
  full_path = './test.avi'
  start = 33
  end = 46

  if os.path.exists('tmp.mp4'):
    os.system('rm tmp.mp4')

  youtube = YouTube("https://www.youtube.com/watch?v="+video_id)
  youtube.set_filename('tmp')


  video = youtube.get('mp4', '360p')

  video.download('.')

  cap = cv2.VideoCapture( 'tmp.mp4' )
  fps = cap.get(cv.CV_CAP_PROP_FPS)
  fourcc = int(cap.get(cv.CV_FOURCC(*'XVID')))
  w = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

  out = cv2.VideoWriter( full_path, fourcc, fps, (w,h))

  start_frame = int(fps * start)
  end_frame = int(fps * end)

  frame_count = 0
  while frame_count < end_frame:
    ret, frame = cap.read()
    frame_count += 1

    if frame_count >= start_frame:
        out.write(frame)

  cap2 = cv2.VideoCapture (full_path)
  ret, frame = cap2.read()

  if ret is False:
    print "ret is False"
  else:
    print "ret is True"

if __name__=="__main__":
    main()


