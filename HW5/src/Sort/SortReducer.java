package page_rank;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class SortReducer extends Reducer<SortPair, DoubleWritable, Text, DoubleWritable> {

    public void reduce(SortPair key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        for (DoubleWritable val: values) {
            context.write(key.getKey(), val);
        }
    }
}
