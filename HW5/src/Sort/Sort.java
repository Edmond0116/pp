package page_rank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;

public class Sort {
    
    public Sort() {}

    public void Sort(String input, String output) throws Exception {
        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf, "Sort");
        job.setJarByClass(Sort.class);

        // set the inputFormatClass <K, V>
        job.setInputFormatClass(KeyValueTextInputFormat.class);

        // set the class of each stage in mapreduce
        job.setMapperClass(SortMapper.class);
        //job.setPartitionerClass(SortPartitioner.class);
        job.setSortComparatorClass(SortKeyComparator.class);
        job.setGroupingComparatorClass(SortGroupComparator.class);
        job.setReducerClass(SortReducer.class);

        // set the output class of Mapper and Reducer
        job.setMapOutputKeyClass(SortPair.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        // set the number of reducer
        job.setNumReduceTasks(1);

        // add input/output path
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.waitForCompletion(true);
    }
}
