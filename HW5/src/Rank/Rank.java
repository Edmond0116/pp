package page_rank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;

public class Rank {
    
    public Rank() {}

    public void Rank(int R, int N, double sum, String input, String output) throws Exception {
        // param = (1 - a) * (1 / N) + a * sum(dj / N)
        double param = 0.15 * (1. / N) + 0.85 * sum;
        Configuration conf = new Configuration();
        conf.set("param", Double.toString(param));

        Job job = Job.getInstance(conf, "Rank");
        job.setJarByClass(Rank.class);

        // set the inputFormatClass <K, V>
        job.setInputFormatClass(KeyValueTextInputFormat.class);

        // set the class of each stage in mapreduce
        job.setMapperClass(RankMapper.class);
        //job.setCombinerClass(RankCombiner.class);
        job.setReducerClass(RankReducer.class);

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
    }
}
