package page_rank;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

public class SortPartitioner extends Partitioner<SortPair, DoubleWritable> {
    @Override
    public int getPartition(SortPair key, DoubleWritable value, int numReduceTasks) {
        double code = Math.log1p(Double.parseDouble(value.toString())) / Math.log1p(1);
        return (int) (code * numReduceTasks);
        //return (int) (Double.parseDouble(value.toString()) * numReduceTasks);
        //return (key.getVal().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
    }
}
