package page_rank;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;

public class ConvReducer extends Reducer<NullWritable, DoubleWritable, NullWritable, DoubleWritable> {

    public void reduce(NullWritable key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        double ret = 0;
        for (DoubleWritable val : values)
            ret += Double.parseDouble(val.toString());
        context.write(NullWritable.get(), new DoubleWritable(ret));
    }
}
