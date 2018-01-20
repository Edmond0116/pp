package page_rank;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;

public class Graph {

    public Graph() {}

    public void Graph(int R, int N, String input, String output) throws Exception {
        Configuration conf = new Configuration();
        conf.set("N", Integer.toString(N));

        Job job = Job.getInstance(conf, "Graph");
        job.setJarByClass(Graph.class);

        // set the inputFormatClass <K, V>
        job.setInputFormatClass(KeyValueTextInputFormat.class);

        // set the class of each stage in mapreduce
        // Identity Map
        //job.setCombinerClass(GraphCombiner.class);
        job.setReducerClass(GraphReducer.class);

        // set the output class of Mapper and Reducer
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // set the number of reducer
        job.setNumReduceTasks(R);

        // add input/output path
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.waitForCompletion(true);
    }
}
