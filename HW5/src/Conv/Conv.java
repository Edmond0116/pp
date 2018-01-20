package page_rank;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;

public class Conv {

    private double mConv = .0;

    public Conv() {}

    public void Conv(int R, String input, String output) throws Exception {
        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf, "Conv");
        job.setJarByClass(Conv.class);

        // set the inputFormatClass <K, V>
        job.setInputFormatClass(KeyValueTextInputFormat.class);

        // set the class of each stage in mapreduce
        job.setMapperClass(ConvMapper.class);
        job.setCombinerClass(SumReducer.class);
        job.setReducerClass(SumReducer.class);

        // set the output class of Mapper and Reducer
        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(DoubleWritable.class);

        // set the number of reducer
        job.setNumReduceTasks(R);

        // add input/output path
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.waitForCompletion(true);

        mConv = readConvDouble(output);
    }

    public double getConv() { return mConv; }

    private double readConvDouble(String input) throws IOException {
        FileSystem fs = FileSystem.get(new Configuration());
        FileStatus[] status = fs.listStatus(new Path(input));
        double ret = 0;
        for (FileStatus f : status) {
            if (!f.isFile()) continue;
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(f.getPath())));
            String res = reader.readLine();
            if (res != null) ret += Double.parseDouble(res);
            reader.close();
        }
        return ret;
    }
}
