package page_rank;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class SortMapper extends Mapper<Text, Text, SortPair, DoubleWritable> {
    
    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        if (key.toString().equals("epsilon")) return;
        DoubleWritable val = new DoubleWritable(Double.parseDouble(value.toString().split("\t", 2)[0]));
        context.write(new SortPair(key, val), val);
    }
}

