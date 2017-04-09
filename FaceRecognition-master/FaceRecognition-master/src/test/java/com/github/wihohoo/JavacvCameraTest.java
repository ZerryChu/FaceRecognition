/** 
 * 文件名：javavcCameraTest.java 
 * 描述：调用windows平台的摄像头窗口视频 
 * 修改时间：2016年6月13日 
 * 修改内容： 
 */
package com.github.wihohoo;

import javax.swing.JFrame;

import org.bytedeco.javacv.*;

import com.github.wihoho.Trainer;
import com.github.wihoho.constant.FeatureType;
import com.github.wihoho.jama.util.MatrixUtil;
import com.github.wihoho.training.CosineDissimilarity;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.javacpp.opencv_core.IplImage;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

import java.io.File;
import java.io.IOException;

/**
 * 调用本地摄像头窗口视频
 * 
 * @author eguid
 * @version 2016年6月13日
 * @see javavcCameraTest
 * @since javacv1.2
 */
public class JavacvCameraTest {
	final static boolean impassabled = false;
	private static CvHaarClassifierCascade classifier;
	private static MatrixUtil matrixUtil = new MatrixUtil();
	private static Trainer trainer = new Trainer();
	private static CvFont cvFont = new CvFont();
	private static int newWidth = 92;
	private static int newHeight = 112;
	
	public JavacvCameraTest() {
		// TODO Auto-generated constructor stub
		// Load the classifier file from Java resources.
		try {
			Loader.load(opencv_objdetect.class);
			classifier = new CvHaarClassifierCascade(cvLoad(
				"/Users/zhuzirui/Downloads/FaceRecognition-master/src/test/resources/haarcascade_frontalface_alt2.xml"));
		
			train();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void train() throws Exception {
		// Build a trainer
        trainer.metric = new CosineDissimilarity();
        trainer.featureType = FeatureType.PCA;
        trainer.numberOfComponents = 3;
        trainer.k = 1;
        
        String john1 = "faces/s2/1.pgm";
        String john2 = "faces/s2/2.pgm";
        String john3 = "faces/s2/3.pgm";
        String john4 = "faces/s2/4.pgm";

        String smith1 = "faces/s4/1.pgm";
        String smith2 = "faces/s4/2.pgm";
        String smith3 = "faces/s4/3.pgm";
        String smith4 = "faces/s4/4.pgm";

        String zerry1 = "faces/myface/1.pgm";
        String zerry2 = "faces/myface/2.pgm";
        String zerry3 = "faces/myface/3.pgm";
        String zerry4 = "faces/myface/4.pgm";
        
        // add training data
        trainer.add(matrixUtil.convertToMatrix(smith1), "smith");
        trainer.add(matrixUtil.convertToMatrix(smith2), "smith");
        trainer.add(matrixUtil.convertToMatrix(smith3), "smith");
        trainer.add(matrixUtil.convertToMatrix(smith4), "smith");
        
        trainer.add(matrixUtil.convertToMatrix(john1), "john");
        trainer.add(matrixUtil.convertToMatrix(john2), "john");
        trainer.add(matrixUtil.convertToMatrix(john3), "john");
        trainer.add(matrixUtil.convertToMatrix(john4), "john");

        trainer.add(matrixUtil.convertToMatrix(zerry1), "zerry");
        trainer.add(matrixUtil.convertToMatrix(zerry2), "zerry");
        trainer.add(matrixUtil.convertToMatrix(zerry3), "zerry");
        trainer.add(matrixUtil.convertToMatrix(zerry4), "zerry");
    
        // train
        trainer.train();
	}
	
	public static String detect(String target_address) throws IOException {
		return trainer.recognize(matrixUtil.convertToMatrix(target_address));
	}
	
	public static IplImage resizeImage(IplImage grayImage) {
		IplImage target = cvCreateImage(cvSize(newWidth, newHeight), grayImage.depth(), grayImage.nChannels());
		// cvResetImageROI((IplImage) grayImage);
		if (newWidth > grayImage.width() && newHeight > grayImage.height()) {
			// Make the image larger
			cvResize(grayImage, target, CV_INTER_LINEAR); // CV_INTER_CUBIC or
													   	  // CV_INTER_LINEAR is
														  // good for enlarging
		} else {
			// Make the image smaller
			cvResize(grayImage, target, CV_INTER_AREA); // CV_INTER_AREA is good
														// for shrinking /
														// decimation, but bad
														// at enlarging.
		}
		opencv_imgcodecs.cvSaveImage("target.pgm", target);
		return target;
	}
	
	public static void main(String[] args) throws Exception, InterruptedException {
		new JavacvCameraTest();
		OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
		grabber.start(); // 开始获取摄像头数据
		CanvasFrame canvas = new CanvasFrame("camera");// 新建一个窗口
		canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		canvas.setAlwaysOnTop(true);
		CvMemStorage storage = CvMemStorage.create();
		cvInitFont(cvFont, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 1, 2, 8);
                
		while (true) {
			if (!canvas.isDisplayable()) {// 窗口是否关闭
				grabber.stop();// 停止抓取
				System.exit(2);// 退出
			}
			Frame frame = grabber.grab();
			//canvas.showImage(frame); // 获取摄像头图像并放到窗口上显示， 这里的Frame
										// frame=grabber.grab();
										// frame是一帧视频图像

			OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
			IplImage image = null;
			IplImage grayImage = null;
			// IplImage rotatedImage = null;
		    // CvPoint hatPoints = new CvPoint(3);
			// CvMat randomR = CvMat.create(3, 3), randomAxis = CvMat.create(3, 1);

			if (frame != null) {
				image = converter.convert(frame);
				int height = image.height();
				int width = image.width();
				grayImage = IplImage.create(width, height, IPL_DEPTH_8U, 1);
				// rotatedImage = image.clone();

				 //opencv_imgcodecs.cvSaveImage("hello.pgm", image);
				cvClearMemStorage(storage);
				cvCvtColor(image, grayImage, CV_BGR2GRAY);

				//*** 选择最大的face进行识别，排除错误
				CvSeq faces = cvHaarDetectObjects(grayImage, classifier, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING);
				int total = faces.total();
				CvRect r = null;
				CvRect temp = null;
				for (int i = 0; i < total; i++) {
					temp = new CvRect(cvGetSeqElem(faces, i));
					int w = temp.width(), h = temp.height();
					if (w < 100 && h < 100)
						continue;
					if (r == null || (w > r.width() && h > r.height()) ) {
						r = temp;
					}
				}

				if (image != null && r != null) {
					int x = r.x(), y = r.y(), w = r.width(), h = r.height();
					cvRectangle(image, cvPoint(x, y), cvPoint(x + w, y + h), CvScalar.RED, 1, CV_AA, 0);
				
					/* 帽子
					hatPoints.position(0).x(x - w / 10).y(y - h / 10);
					hatPoints.position(1).x(x + w * 11 / 10).y(y - h / 10);
					hatPoints.position(2).x(x + w / 2).y(y - h / 2);
					cvFillConvexPoly(image, hatPoints.position(0), 3, CvScalar.BLUE, CV_AA, 0);
					*/
					
					// 切割出target.pgm
					cvSetImageROI(grayImage, r);
					IplImage targetImg = resizeImage(grayImage);
					String person = detect("target.pgm");
					cvPutText(image, person, cvPoint(r.x(), r.y() + r.height() + 10), cvFont, CvScalar.RED);  
					/*
					if(g_confidence*100>50){  
		                cvPutText(src, textName,cvPoint(r.x()-10, r.y() + r.height() + 20), font, CvScalar.WHITE); 
		                cvPutText(src, " conf="+Integer.valueOf((int) (g_confidence*100))+"%",cvPoint(r.x()-10, r.y() + r.height() + 40), font, CvScalar.GREEN);  
		                textName="unknow";  
		            }  
		            else {  
						cvPutText(image, person, cvPoint(r.x(), r.y() + r.height() + 10), cvFont, CvScalar.RED);  
		                cvPutText(image, " conf="+Integer.valueOf((int) (g_confidence*100))+"%", cvPoint(r.x()-10, r.y() + r.height() + 40), font, CvScalar.GREEN);  
		            }
		            */				
					
					/* 轮廓
					cvThreshold(grayImage, grayImage, 64, 255, CV_THRESH_BINARY);
	
					CvSeq contour = new CvSeq(null);
					cvFindContours(grayImage, storage, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST,
							CV_CHAIN_APPROX_SIMPLE);
					while (contour != null && !contour.isNull()) {
						if (contour.elem_size() > 0) {
							CvSeq points = cvApproxPoly(contour, Loader.sizeof(CvContour.class), storage, CV_POLY_APPROX_DP,
									cvContourPerimeter(contour) * 0.02, 0);
							cvDrawContours(image, points, CvScalar.BLUE, CvScalar.BLUE, -1, 1, CV_AA);
						}
						contour = contour.h_next();
					}
					*/

				// cvWarpPerspective(image, rotatedImage, randomR);
				
				}
				Frame target = converter.convert(image);
				canvas.showImage(target);

			} else {
				canvas.dispose();
				System.exit(3);// 退出
			}
			// hatPoints.close();
			Thread.sleep(10);// 50毫秒刷新一次图像
		}
	}
	
}