package page_rank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.CounterGroup;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Parse {

    private int mCount = 0;

    public Parse() {}

    public void Parse(int R, String input, String output) throws Exception {
        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf, "Parse");
        job.setJarByClass(Parse.class);

        // set the class of each stage in mapreduce
        job.setMapperClass(ParseMapper.class);
        //job.setCombinerClass(ParseCombiner.class);
        job.setReducerClass(ParseReducer.class);

        // set the output class of Mapper and Reducer
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // set the number of reducer
        job.setNumReduceTasks(R);

        // add input/output path
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.waitForCompletion(true);

        CounterGroup counters = job.getCounters().getGroup("Nodes");
        for (Counter counter : counters)
            mCount += counter.getValue();
    }

    public int getN() { return mCount; }
}
