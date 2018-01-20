package page_rank;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class RankReducer extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        Configuration config = context.getConfiguration();
        String list = "";
        double sum = 0;
        for (Text value : values) {
            String val = value.toString();
            if (val.startsWith("#"))
                list = val.substring(1);
            else
                sum += Double.parseDouble(val);
        }
        // pr = param + a * sum(pr ti / deg ti)
        double pr = Double.parseDouble(config.get("param")) + 0.85 * sum;
        String[] parts = list.split("\t", 2);
        double conv = Math.abs(Double.parseDouble(parts[0]) - pr);
        parts[0] = Double.toString(pr);
        context.write(new Text("epsilon"), new Text(Double.toString(conv)));
        context.write(key, new Text(String.join("\t", parts)));
    }
}
