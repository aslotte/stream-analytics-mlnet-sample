using Microsoft.ML.Data;

namespace MachineLearning.DataModels
{
    public class FraudPrediction
    {
        [ColumnName("PredictionLabel")]
        public bool IsFraudulent { get; set; }
        public float Score { get; set; }
    }
}
