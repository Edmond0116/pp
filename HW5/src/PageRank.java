package page_rank;

import org.apache.hadoop.mapreduce.Job;

public class PageRank {

    public static void main(String[] args) throws Exception {
        /*
           InputFile : args[0]
           OutputDir : args[1] 
           Number of iterations : args[2] */
        // Number of Reducers
        final int R = (args.length >= 4 ? Integer.parseInt(args[3]) : 32);

        // Parse Pages
        Parse parseJob = new Parse();
        parseJob.Parse(R, args[0], args[1] + "/parse");
        final int N = parseJob.getN();

        // Build Graph
        Graph graphJob = new Graph();
        graphJob.Graph(R, N, args[1] + "/parse", args[1] + "/iter/0");

        String sumFolder = args[1] + "/sum/",
            rankFolder = args[1] + "/iter/",
            sortFolder = args[1] + "/sort/",
            convFolder = args[1] + "/conv/";
        double rate = N;
        // Number of Iterations
        final int iters = (args.length >= 3 ? Integer.parseInt(args[2]) : -1);
        int i;
        for (i = 1; iters == -1 ? !(rate < 0.001) : i <= iters; ++i) {
            String prevSum = sumFolder + Integer.toString(i - 1),
                prevRank = rankFolder + Integer.toString(i - 1),
                curRank = rankFolder + Integer.toString(i),
                curSort = sortFolder + Integer.toString(i),
                curConv = convFolder + Integer.toString(i);
            // Sum Dangling
            Sum sumJob = new Sum();
            sumJob.Sum(R, N, prevRank, prevSum);
            double sum = sumJob.getSum();

            // Page Rank
            Rank rankJob = new Rank();
            rankJob.Rank(R, N, sum, prevRank, curRank);

            // Sum Converge
            if (iters == -1) {
                Conv convJob = new Conv();
                convJob.Conv(R, curRank, curConv);
                rate = convJob.getConv();
            }

            // Sort Result
            //Sort sortJob = new Sort();
            //sortJob.Sort(curRank, curSort);
        }

        // Sort Result
        Sort sortJob = new Sort();
        sortJob.Sort(rankFolder + Integer.toString(i - 1), args[1] + "/sort");

        System.exit(0);
    }
}

