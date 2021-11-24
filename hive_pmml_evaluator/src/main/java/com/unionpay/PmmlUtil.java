package com.unionpay;

import org.jpmml.evaluator.LoadingModelEvaluatorBuilder;
import org.xml.sax.SAXException;
import java.io.File;
import java.io.IOException;
import javax.xml.bind.JAXBException;
import org.jpmml.evaluator.Evaluator;
import org.apache.hadoop.conf.Configuration;
import java.io.InputStream;

import java.io.EOFException;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;

public class PmmlUtil {
    private static Evaluator model = null;

    public static Evaluator getEvaluator(){
        return model; 
    }

    public static void loadModel(String PmmlFile) throws IOException, SAXException, JAXBException {
        Path path = new Path(PmmlFile);
        InputStream in = getInputStream(path,new Configuration());
        //model = new LoadingModelEvaluatorBuilder().load(new File(PmmlFile)).build();    
        model = new LoadingModelEvaluatorBuilder().load(in).build();    
    }

    public static InputStream getInputStream(Path filename, Configuration conf) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream i = fs.open(filename);
        FileStatus stats = fs.getFileStatus(filename);
  
        short leadBytes;
        try {
           leadBytes = i.readShort();
        } catch (EOFException var9) {
           i.seek(0L);
           return i;
        }
  
        CompressionCodecFactory factory = new CompressionCodecFactory(conf);
        CompressionCodec codec = factory.getCodec(filename);
        if (codec != null) {
            i.seek(0L);
            return codec.createInputStream(i);
        }

  
        i.seek(0L);
        return i;
     }


}

