package page_rank;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class ConvMapper extends Mapper<Text, Text, NullWritable, DoubleWritable> {
    
    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        if (key.toString().equals("epsilon"))
            context.write(NullWritable.get(), new DoubleWritable(Double.parseDouble(value.toString())));
    }
}

