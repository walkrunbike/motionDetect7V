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

class Camera:
   def __init__(self):
      config = ConfigManager()
      self.videoURL = config.get("Camera", "cameraURL")    
      self.timeout  = int(config.get("Camera", "socketTimeout")) 

      # Image source type can be: 'MJPEG'
      self.imgSourceType  = config.get("Camera", "cameraFormat")
      self.mjpegStream    = ''
      self.capFile        = ''      
      self.frameCount     = 0
      self.frameTime      = 0
      socket.setdefaulttimeout(self.timeout)
      self.setImageSource(self.imgSourceType)

   def getFrame(self):
      img = self.getMjpegFrame()

      if self.capFile != "":
         # shouldn't capture data again if already from a file
         if not isinstance(self.mjpegStream, file):
            self.capFile.write("Content-Length:%d\n" % len(img))
            self.capFile.write("\n")
            self.capFile.write(img)
            
      # convert to np array
      data = np.asarray(bytearray(img), dtype="uint8")
      #cvImage = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_COLOR)
      #3.0
      cvImage = cv2.imdecode(data, cv2.IMREAD_COLOR)
      
      
      self.frameCount += 1
      if self.frameCount == 1:
         self.frameTime = time.time()
         
      delta = time.time() - self.frameTime
      if delta > 60:
         Log( "Frames Retrieved Per Sec = %0.1f" % (self.frameCount / delta) )
         self.frameCount = 0
      return cvImage
   
   def saveFrame(self, img, fileName):
      f = open(saveFileName, "wb")
      f.write(data)
      f.close()   

   def setImageSource(self, sourceType = "MJPEG", sourceLocation = ""):
      self.imgSourceType = sourceType
      if sourceType == "MJPEG":
         self.mjpegStream = self.__openMjpegStream()
      elif sourceType == "FILE":
         self.mjpegStream = open(sourceLocation, "rb")
      else:
         self.mjpegStream = ""

   def __openMjpegStream(self):
      """
      open connection for mjpeg stream
      exception if connection cannot be established
      """
      stream = ""
      attempt = 0
      while attempt < 5 and stream == "":
         try:
            stream = urllib2.urlopen(self.videoURL)
         except:
            time.sleep(1)
            attempt += 1
      return stream
            
   def __readMjpegStream(self, bytes = 0):
      """ read data from the stream
          data can be captured to a file, if capFile handle is set
      """
      data = ""
      if bytes == 0:
         data = self.mjpegStream.readline()
         # allow looping if data is coming from a file
         if len(data) == 0 and isinstance(self.mjpegStream, file):
            self.mjpegStream.seek(0,0)
      else:
         data = self.mjpegStream.read(bytes)
      return data
      
   def getMjpegFrame(self):
      data = ""
      while data.find("Content-Length:") < 0: 
         data = self.__readMjpegStream()
      bytes = int(data.replace("Content-Length:", ""))
      self.__readMjpegStream()
      data = self.__readMjpegStream(bytes)
      return data
      
   def startCapture(self, fileName = ""):
      self.capFile = open(fileName, "ab")
      pass
   
   def stopCapture(self):
      if self.capFile != "":
         self.capFile.close()
         self.capFile = ""


class MotionDetector:
   def __init__(self):
      config = ConfigManager()
      x,y,w,h = (config.get("Detection", "preferredArea")).split(",")
      self.preferredArea = (int(x), int(y), int(w), int(h))
      self.camera = Camera()
      self.viewController = ViewController(self.camera)
      self.archiver = Archiver(640, 480)
      self.objectHistory = []
   
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
      #contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      #3.0
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
         
   def isMotionInPreferredArea(self, rectangle):
      (px, py, pw, ph) = self.preferredArea
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
         activityInPreferredArea = False
         if self.isMotionInPreferredArea( rectInMotion ):
            activityInPreferredArea = True
            Log( "Event detected: (%d, %d, %d, %d)" % rectInMotion )
            self.drawRectangles(img1, [rectInMotion], (0,0,255))            
         else:
            activityInPreferredArea = False            
            Log( "Motion detected: (%d, %d, %d, %d)" % rectInMotion )            
            self.drawRectangles(img1, [rectInMotion], (255, 0, 0))
         self.archiver.saveImage(img1, activityInPreferredArea)
         self.archiver.saveImage(img2, activityInPreferredArea)
      else:
         self.archiver.checkSessionEnded()
      self.drawRectangles(img1, [self.preferredArea], (0, 255, 0) )   
      self.viewController.display(img1, imgDiff)
      
   def startDetect(self):
      Log( "Starting Detection" )
      running = True
      while running:
         running = self.viewController.isRunning()
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
      
      #fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
      #3.0
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
      videoName = "capture_%s.mp4" % datetime.datetime.now().strftime("%Y%m%d_%H%M")
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
         vidName = "capture_%s.avi" % datetime.datetime.now().strftime("%Y%m%d_%H%M")
         self.ftp.storbinary("STOR " + videoName, open(videoName, "rb"), 1024)
         self.ftp.quit()
         Log( "Video Uploaded: %s" % vidName )
 
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
      pass

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
         self.camera.startCapture('capture.mjpg')
      # x = exit capture
      elif key & 0xFF == ord('x'):
         self.camera.stopCapture()
      # r = read from file
      elif key & 0xFF == ord('r'):
         self.camera.setImageSource('FILE', 'capture.mjpg')
      # c = read from camera
      elif key & 0xFF == ord('c'):
         self.camera.setImageSource('MJPEG')
      # o = one frame
      elif key & 0xFF == ord('o'):
         self.oneFrame = True
      # m = multi frame
      elif key & 0xFF == ord('m') and self.oneFrame == True:
         self.oneFrame = False   
      return keepRunning

   def display(self, actualImage, diffImage):
      if self.enableDisplay == "False":
         return
         
      if self.displayType == "actual":
         cv2.imshow("motionDetect7V", actualImage)
         self.lastImage = actualImage
      elif self.displayType == "diff":
         cv2.imshow("motionDetect7V", diffImage)
         self.lastImage = diffImage         
         
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
   
def main():
   isRunning = True
   while isRunning:
      try:
         detector = MotionDetector()
         isRunning = detector.startDetect()
      except KeyboardInterrupt:
         Log( "Exiting - Keyboard Interrupt" )
         exit()
      except FatalException as f:
         Log( f )
         exit()
      except:
         Log( traceback.format_exc() )
      
if __name__ == "__main__":
   main()
