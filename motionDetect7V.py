import urllib2
import socket
import numpy as np
import cv2
import time
import ftplib
import smtplib
import datetime
import os
import os.path
import ConfigParser
import subprocess
import StringIO
import datetime
import traceback
import multiprocessing


class VideoSource(object):
   def __init__(self):
      self.frameCount  = 0
      self.frameTimer  = 0
      self.capFile     = ""
   
   def getFrame(self):
      self.frameCount += 1
      if self.frameCount == 1:
         self.frameTimer = time.time()
         
      delta = time.time() - self.frameTimer
      if delta > 60:
         fps = (self.frameCount / delta)
         Log( "Frames Retrieved Per Sec = %0.1f" % fps )
         self.frameCount = 0
         if (fps < 4.0):
            raise NonFatalException("Low Frame Count Exception")       
      pass
   
   def saveFrame(self, img):
      # todo: need to be jpeg
      if self.capFile != "":
         r, output = cv2.imencode(".jpg", img)
         self.capFile.write("Content-Length:%d\n" % len(output))
         self.capFile.write("\n")
         self.capFile.write(output)
   
   def startCaptureToFile(self, fileName = ""):
      self.capFile = open(fileName, "ab")
   
   def stopCaptureToFile(self):
      if self.capFile != "":
         self.capFile.close()
         self.capFile = ""

class CameraMJPEG(VideoSource):
   def __init__(self):
      """
      open connection for mjpeg stream                            
      exception if connection cannot be established
      """
      super(self.__class__, self).__init__()
      config = ConfigManager()
      self.videoURL = config.get("Camera", "cameraURL")
      self.timeout  = int(config.get("Camera", "socketTimeout"))
      socket.setdefaulttimeout(self.timeout)      
      stream = ""
      attempt = 0
      while attempt < 5 and stream == "":
         try:
            stream = urllib2.urlopen(self.videoURL)
         except:
            time.sleep(1)
            attempt += 1
      if stream == "":
         raise NonFatalException("Unable to open camera URL")
      self.videoStream = stream

   def getFrame(self):
      super(CameraMJPEG, self).getFrame()      
      data = ""
      while data.find("Content-Length:") < 0: 
         data = self.__readMjpegStream()
      bytes = int(data.replace("Content-Length:", ""))
      self.__readMjpegStream()
      data = self.__readMjpegStream(bytes)
   
      # convert to np array
      cVImage = ""
      data = np.asarray(bytearray(data), dtype="uint8")
      # cv2 version differences         
      if (cv2.__version__[0] == "2"):
         cvImage = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_COLOR)
      else:
         cvImage = cv2.imdecode(data, cv2.IMREAD_COLOR)
      super(CameraMJPEG, self).saveFrame(cvImage)          
      return cvImage                                                        
      
   def __readMjpegStream(self, bytes = 0):
      """ read data from the stream
      """
      data = ""
      if bytes == 0:
         data = self.videoStream.readline()
      else:
         data = self.videoStream.read(bytes)
      return data
      

class CameraRTSP(VideoSource):
   def __init__(self):
      super(self.__class__, self).__init__()
      config = ConfigManager()
      self.videoURL = config.get("Camera", "cameraURL")
      self.timeout  = int(config.get("Camera", "socketTimeout"))       
      self.videoStream = cv2.VideoCapture(self.videoURL)
      socket.setdefaulttimeout(self.timeout)

   def getFrame(self):
      super(CameraRTSP, self).getFrame()      
      r, cvImage = self.videoStream.read()
      h,w = cvImage.shape[:2]
      if (h != 480) or (w != 640):
         Log("Warning: 480x640 expected. Image dimension = %d x %d" % (h, w))
      super(CameraRTSP, self).saveFrame(cvImage) 
      return cvImage
      
   
class CameraFileCapture(VideoSource):
   def __init__(self, sourceFileName):
      self.videoStream = open(sourceFileName, "rb")

   def getFrame(self):
      data = ""
      while data.find("Content-Length:") < 0: 
         data = self.__readMjpegStream()
      bytes = int(data.replace("Content-Length:", ""))
      self.__readMjpegStream()
      data = self.__readMjpegStream(bytes)

      # convert to np array
      cVImage = ""
      data = np.asarray(bytearray(data), dtype="uint8")
      # cv2 version differences         
      if (cv2.__version__[0] == "2"):
         cvImage = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_COLOR)
      else:
         cvImage = cv2.imdecode(data, cv2.IMREAD_COLOR)
      return cvImage
      
   def __readMjpegStream(self, bytes = 0):
      """ read data from the stream
      """
      data = ""
      if bytes == 0:
         data = self.videoStream.readline()
         # allow looping
         if len(data) == 0:
            self.videoStream.seek(0,0)
      else:
         data = self.videoStream.read(bytes)
      return data
      
   
class VideoSourceFactory:
   def __init__(self):
      config = ConfigManager()
      self.videoURL = config.get("Camera", "cameraURL")
      # Image source type can be: 'MJPEG', 'RTSP', 'FILE'
      self.imgSourceType  = config.get("Camera", "cameraFormat")
      
   def getVideoSource(self):         
      if self.imgSourceType == "RTSP":
         videoStream = CameraRTSP() 
      elif self.imgSourceType == "MJPEG":
         videoStream = CameraMJPEG()
      elif self.imgSourceType == "FILE":
         videoStream = CameraFileCapture( 'capture.mjpg' )
      else:
         videoStream = ""
         raise FatalException( "Unknown cameraFormat: %s" % self.imgSourceType )
      return videoStream
      
   def getFileSource(self, sourceName):
      videoStream = CameraFileCapture( sourceName )
      return videoStream

class MotionDetector:
   def __init__(self):
      config = ConfigManager()
      x,y,w,h = (config.get("Detection", "AlertArea")).split(",")
      self.AlertArea = (int(x), int(y), int(w), int(h))
      self.camera = VideoSourceFactory().getVideoSource()
      self.viewController = ViewController(self.camera)
      self.archiver = Archiver(640, 480)
      self.objectHistory = []
      self.CaptureBoarderActivity = config.get("Detection", "CaptureBoarderActivity")
   
   def diffImg2(self, img1, img2):   
      gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
      diff = cv2.absdiff(gray1, gray2)
      return diff
      
   def diffImg3(self, img1, img2, img3):     
      gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
      gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)    
      diff1 = cv2.absdiff(gray1, gray2)
      diff2 = cv2.absdiff(gray2, gray3 )
      return cv2.bitwise_and(diff1, diff2)

   def detectObjectsContours(self, grayScaleImg, colorSensitivityThresh = 10, sizeThresh = 250):
      """ detect objects in image and returning their contours
      """
      retval, threshImg = cv2.threshold(grayScaleImg, colorSensitivityThresh, 255, cv2.THRESH_BINARY)
      threshImg = cv2.dilate(threshImg, None, iterations=2)
      img, contour, hierarchy = (0, 0, 0)
      # cv2 version
      if cv2.__version__[0] == "2":
         contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      else:
         img, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      foundObjects = []
      for c in contours:
         if cv2.contourArea(c) > sizeThresh:
            foundObjects.append(c)
      return foundObjects
      
   def convertContoursToRectangles(self, contours):
      rectangles = []
      for c in contours:
         rectangles.append(cv2.boundingRect(c))
      return rectangles

   def detectRectangleMotion(self, rectList):
      foundObjectInMotion = False
      maxRect = (0, 0, 0, 0)
      maxSize = 0
      # find the larget rect to work with
      for r in rectList:
         (x, y, w, h) = r
         if w * h > maxSize:
            maxSize = w * h
            maxRect = r
      self.objectHistory.append(maxRect)
      
      # Check history of object
      historySize = 3
      if len(self.objectHistory) > historySize:
         foundObjectInMotion = True         
         (x, y, w, h) = self.objectHistory[0]
        
         for r in self.objectHistory:
            (x2, y2, w2, h2) = r
            if w2 == 0:
               foundObjectInMotion = False
               break
            # see if object has moved too far/fast
            if (x2 < (x-w)) or ((x2 + w2) > (x+2*w)):
               foundObjectInMotion = False
               break
               
         self.objectHistory.pop(0)
      return foundObjectInMotion, maxRect
      
   def drawRectangles(self, img, rectangles, color = (255,0,0) ):
      rectThickness = 1
      for r in rectangles:
         (x, y, w, h) = r
         cv2.rectangle(img, (x, y), (x + w, y + h), color, rectThickness)
         
   def isMotionInAlertArea(self, rectangle):
      (px, py, pw, ph) = self.AlertArea
      (x, y, w, h) = rectangle
      result = False
      if (x >= px) and ((x+w) <= (px + pw)) and (y >= py) and ((y+h) <= (py + ph)):
         result = True
      return result
   
   def detectNight(self):
      img1 = self.camera.getFrame()
      img2 = self.camera.getFrame()
      imgDiff = self.diffImg2(img1, img2)
      detectedContours = self.detectObjectsContours(imgDiff, 10, 250)
      rects = self.convertContoursToRectangles(detectedContours)
      rectMotionDetected, rectInMotion = self.detectRectangleMotion(rects)
      if rectMotionDetected:
         activityInAlertArea = False
         if self.isMotionInAlertArea( rectInMotion ):
            activityInAlertArea = True
            Log( "Event detected: (%d, %d, %d, %d)" % rectInMotion )
            self.drawRectangles(img1, [rectInMotion], (0,0,255))            
         else:
            activityInAlertArea = False            
            Log( "Motion detected: (%d, %d, %d, %d)" % rectInMotion )            
            self.drawRectangles(img1, [rectInMotion], (255, 0, 0))
         
         if ((activityInAlertArea == True) or (self.CaptureBoarderActivity == True)): 
            self.archiver.saveImage(img1, activityInAlertArea)
            self.archiver.saveImage(img2, activityInAlertArea)
      else:
         self.archiver.checkSessionEnded()
      self.drawRectangles(img1, [self.AlertArea], (0, 255, 0) )   
      self.viewController.display(img1, imgDiff)
      
   def doDetection(self):
      # there should be a better way to update camera setting
      running, self.camera = self.viewController.isRunning()
      self.detectNight()
      return running


class Archiver:
   def __init__(self, width, height):
      config = ConfigManager()
      self.ftpEnable  = config.get("NetworkStorage", "ftpEnable")
      self.ftpAddress = config.get("NetworkStorage", "ftpIPAddr")
      self.ftpUser    = config.get("NetworkStorage", "ftpUser")
      self.ftpPsw     = config.get("NetworkStorage", "ftpPsw")
      self.ftpDir     = config.get("NetworkStorage", "ftpDirectory")
      self.fileName  = "tmpCapturedEvent.avi"
      self.dimension = (width, height)
      self.frmTimes  = []
      self.framesPerPeriod = 10
      self.minPeriod       = 10
      self.minIdleTime     = 10
      self.startSession()

   def __del__(self):
      self.archiveSession()
      
   def startSession(self):
      # If a file exists from a previous session, archive it
      if (os.path.isfile(self.fileName)):
         self.archiveSession()
      
      fourcc = ""
      # cv2 version differences
      if cv2.__version__[0] == "2":
         fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
      else:
         fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
      self.video = cv2.VideoWriter()
      self.video.open(self.fileName, fourcc, 15, self.dimension)
   
   def saveImage(self, img, sessionActivity = True):
      self.video.write(img)
      if sessionActivity == True:
         self.frmTimes.append( int(time.time()) )
   
   def endSession(self):
      self.archiveSession()
      Emailer().send()
      self.video = ""
      self.startSession()
      pass
   
   def checkSessionEnded(self):
      # Session ends when
      # There been at least X frames saved within a Y second period
      # and no activity has been saved in the last Z seconds
      sessionEnd = False
      currentTime = int(time.time())
      # Need at least X frames
      if len(self.frmTimes) > self.framesPerPeriod:
         # Make sure system is idle - newest frame is over Z seconds ago
         if currentTime >= self.frmTimes[len(self.frmTimes) - 1] + self.minPeriod:
            indexesFrame = range(0, len(self.frmTimes) - 1)
            indexesFrame.reverse()
            for i in indexesFrame:
               # check X frames ago, and compare time to see if they
               # occurred within Y seconds
               if (i - self.framesPerPeriod) > 0:
                  if self.frmTimes[i] - self.frmTimes[i-10] < self.minPeriod:
                     sessionEnd = True
                     self.endSession()
                     break
            self.frmTimes = []
      return sessionEnd
      
   def convertToMP4(self, inFile):
      valid = True
      videoName = "capture_%s.mp4" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      os.system("mencoder -speed 1/2 -ovc lavc -lavcopts vcodec=mpeg4 -of lavf %s -o %s 1>/dev/null 2>&1" % (inFile, videoName))
      Log( "Creating MP4: %s" % videoName)
      os.remove(inFile)
      if (os.stat(videoName).st_size == 0):
         os.remove(videoName)
         Log( "MP4 has no content")
         valid = False
      return valid, videoName      
                                                              
   def archiveSession(self):
      valid, videoName = self.convertToMP4(self.fileName)
      if valid == True and self.ftpEnable == "True":
         self.ftp = ftplib.FTP(self.ftpAddress)
         self.ftp.login( self.ftpUser, self.ftpPsw)
         self.ftp.cwd( self.ftpDir )
         self.ftp.storbinary("STOR " + videoName, open(videoName, "rb"), 1024)
         self.ftp.quit()
         Log( "Video Uploaded: %s" % videoName )
 
class ViewController:
   def __init__(self, theCamera):
      config = ConfigManager()
      self.enableDisplay = config.get("Display", "enableDisplay")
      self.displayType = 'actual'
      self.oneFrame = False
      self.camera = theCamera
      
      self.lastImage = ""
      self.drawing = False
      self.X1, self.Y1, self.X2, self.Y2 = 0,0,0,0
      

   def isRunning(self):
      keepRunning = True
      key = ""
      if self.oneFrame == True:
         key = cv2.waitKey()
      else:
         key = cv2.waitKey(1)
         
      # q = quit
      if key & 0xFF == ord('q'):
         keepRunning = False
      # a = actual image
      elif key & 0xFF == ord('a'):
         self.displayType = 'actual'
      # d = diff image
      elif key & 0xFF == ord('d'):
         self.displayType = 'diff'
      # s = save capture
      elif key & 0xFF == ord('s'):
         self.camera.startCaptureToFile('capture.mjpg')
      # x = exit capture
      elif key & 0xFF == ord('x'):
         self.camera.stopCaptureToFile()
      # r = read from file
      elif key & 0xFF == ord('r'):
         self.camera = VideoSourceFactory().getFileSource('capture.mjpg')         
      # c = read from camera
      elif key & 0xFF == ord('c'):
         self.camera = VideoSourceFactory().getVideoSource()
      # o = one frame
      elif key & 0xFF == ord('o'):
         self.oneFrame = True
      # m = multi frame
      elif key & 0xFF == ord('m') and self.oneFrame == True:
         self.oneFrame = False   
      return keepRunning, self.camera

   def display(self, actualImage, diffImage):
      if self.enableDisplay == "False":
         return
         
      if self.displayType == "actual":
         cv2.imshow("motionDetect7V", actualImage)
         self.lastImage = actualImage
      elif self.displayType == "diff":
         cv2.imshow("motionDetect7V", diffImage)
         self.lastImage = diffImage
      cv2.resizeWindow("motionDetect7V", 480, 640)
         
   def drawMsg(self, img, msg):
      font = cv2.FONT_HERSHEY_SIMPLEX
      fontSize = 0.5
      fontColor = (0,0,255)
      fontThickness = 2
      cv2.putText(img, msg,(500,50), font, fontSize,fontColor,fontThickness)     

class Emailer:
   lastSendTime = 0
   
   def __init__(self):
      config = ConfigManager()
      self.enableEmail = config.get("Email", "enableEmail")
      self.fromAddr = config.get("Email", "fromAddr")
      self.toAddr   = config.get("Email", "toAddr")
      self.subject  = config.get("Email", "subject")
      self.content  = config.get("Email", "content")
      self.username = config.get("Email", "username")
      self.pswd     = config.get("Email", "pswd")
      self.smtpAddr = config.get("Email", "smtpAddr")     
      self.sendTimeLimit = 5 * 60

   def send(self):
      if self.enableEmail == "False":
         Log( "Email not enabled." )
         return
         
      if time.time() - Emailer.lastSendTime > self.sendTimeLimit:
         fromMsg= "From: %s" % self.fromAddr
         toMsg  = "To: %s" % self.toAddr        
         msg = "\r\n".join([fromMsg, toMsg, self.subject, "", self.content])
         server = smtplib.SMTP(self.smtpAddr)
         server.ehlo()
         server.starttls()
         server.login( self.username, self.pswd)
         server.sendmail( self.fromAddr, [self.toAddr], msg)
         server.quit()
         Emailer.lastSendTime = time.time()
         Log( "Email Sent." )
      else:
         Log( "Email Not Sent due time limit." )


class ConfigManager:
   class __singleton:
      def __init__(self):
         self.config = ""
         pass
      
   theInstance = None
   
   def __init__(self):
      if not ConfigManager.theInstance:
         ConfigManager.theInstance = ConfigManager.__singleton()
         self.readConfigFile()
      
   def readConfigFile(self):
      encryptedConf = "motionDetect7V.conf"
      textInitFile  = "motionDetect7V.ini"      
      if os.path.isfile(encryptedConf):
         self.readEncryptedConf(encryptedConf)
      elif os.path.isfile(textInitFile):
         self.readTextIni(textInitFile)
      else:
         FatalExeception( "No motionDetect7V.ini found" )
         exit()

   def readEncryptedConf(self, fileName):
      p = subprocess.Popen(['openssl', 'aes-256-cbc', '-d', '-in', fileName], stdout=subprocess.PIPE, stderr=subprocess.PIPE)   
      confParams, errors = p.communicate()
      self.theInstance.config = ConfigParser.ConfigParser()
      s = StringIO.StringIO(confParams)
      try:
         self.theInstance.config.readfp(s)
      except:
         Log( "Error reading: %s" % fileName)
            
   def readTextIni(self, fileName):
      self.theInstance.config = ConfigParser.ConfigParser()
      try:
         self.theInstance.config.read(fileName)
      except:
         Log( "Error reading: %s" % fileName)      
      
   def get(self, sectionText, paramText):
      val = ""
      try:
         val = ConfigManager.theInstance.config.get(sectionText, paramText)
      except:
         mesg = "Missing parameter in config: '%s' in section '%s'" % (paramText, sectionText) 
         raise FatalException( mesg )
      return val


class FatalException(Exception):
   def __init__(self, value):
      self.value = value
      
   def __str__(self):
      return repr(self.value)

class NonFatalException(Exception):
   def __init__(self, value):
      self.value = value
      
   def __str__(self):
      return repr(self.value)

class Log:
   class __singleton:
      def __init__(self):
         self.fp = open(Log.fileName, "a")
         pass
   
   fileName = "motionDetect7V.log"
   theInstance = None
   
   def __init__(self, msg):
      if not Log.theInstance:
         Log.theInstance = Log.__singleton()
      now = datetime.datetime.now()         
      Log.theInstance.fp.write("%s: %s\n" % (now.strftime("%Y-%m-%d %H:%M"), msg))
      Log.theInstance.fp.flush()
      os.fsync(Log.theInstance.fp)
      print( msg )


def startDetector(mesgQueue):
   detector = MotionDetector()
   isRunning = True
   heartbeatCount = 0
   while isRunning:
      isRunning = detector.doDetection()
      if (not isRunning):
         mesgQueue.put("Exit")
         
      heartbeatCount = heartbeatCount + 1
      if ( heartbeatCount > 100 ):
         mesgQueue.put("Heartbeat")
         heartbeatCount = 0


def main():
   statusMsgQ = multiprocessing.Queue()
   keepAlive = True
   heartbeat = True
    
   while keepAlive:
      # start the motion detector
      p = multiprocessing.Process(target=startDetector, name="motionDetecor", args=(statusMsgQ,))       
      p.start()

      # periodic heartbeat messages are expected from detector
      while heartbeat:
         try:
            status = statusMsgQ.get(timeout=60)
            if status == "Exit":
               heartbeat = False
               keepAlive = False                  
         except KeyboardInterrupt:
            Log( "Exiting - Keyboard Interrupt" )
            heartbeat = False
            keepAlive = False  
         except FatalException as f:
            Log( f )
            heartbeat = False
            keepAlive = False            
         except:
            Log( traceback.format_exc() )
            heartbeat = False
            keepAlive = True            

      p.terminate()
      p.join(10)
    
if __name__ == "__main__":
   main()

