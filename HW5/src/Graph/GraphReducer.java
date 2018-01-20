package page_rank;

import java.io.IOException;
import java.util.StringJoiner;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class GraphReducer extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        int N = Integer.parseInt(context.getConfiguration().get("N"));
        Double PR = 1. / N;
        if (key.toString().equals("#")) {
            for (Text val : values)
                context.write(val, new Text(PR.toString()));
        } else {
            StringJoiner adj = new StringJoiner("\t");
            adj.add(PR.toString());
            for (Text val : values) {
                String title = val.toString();
                if (!title.equals("#"))
                    adj.add(title);
            }
            context.write(key, new Text(adj.toString()));
        }
    }
}
