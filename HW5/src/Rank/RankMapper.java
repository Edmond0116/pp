package page_rank;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class RankMapper extends Mapper<Text, Text, Text, Text> {
    
    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\t");
        int deg = parts.length - 1;
        if (deg > 0) {
            double pr = Double.parseDouble(parts[0]) / deg;
            Text p = new Text(Double.toString(pr));
            for (int i = 0; i < deg; ++i)
                context.write(new Text(parts[i + 1]), p);
        }
        context.write(key, new Text("#" + value.toString()));
    }
}

