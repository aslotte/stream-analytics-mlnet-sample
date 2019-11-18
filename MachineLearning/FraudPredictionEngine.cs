using MachineLearning.DataModels;
using Microsoft.ML;
using System.Linq;
using System.Reflection;

namespace MachineLearning
{
    public static class FraudPredictionEngine
    {
        public static bool IsFraudulent(string paymentType,
            string destinationAccountName,
            double amount,
            double oldBalanceOfOriginAccount,
            double oldBalanceOfDestinationAccount,
            double newBalanceOfOriginAccount,
            double newBalanceOfDestinationAccount)
        {
            var mlContext = new MLContext();

            var transaction = new Transaction
            {
                Type = paymentType,
                Amount = (float)amount,
                OldbalanceOrg = (float)oldBalanceOfOriginAccount,
                OldbalanceDest = (float)oldBalanceOfDestinationAccount,
                NewbalanceDest = (float)newBalanceOfDestinationAccount,
                NewbalanceOrig = (float)newBalanceOfOriginAccount,
                NameDest = destinationAccountName
            };

            var model = LoadModel(mlContext);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<Transaction, FraudPrediction>(model);

            var isFraudulent = predictionEngine.Predict(transaction).IsFraudulent;

            return isFraudulent;
        }

        private static ITransformer LoadModel(MLContext mlContext)
        {
            var assembly = Assembly.GetExecutingAssembly();

            var resource = assembly.GetManifestResourceNames().First(x => x.EndsWith("MLModel.zip"));

            using (var stream = assembly.GetManifestResourceStream(resource))
            {
                return mlContext.Model.Load(stream, out var _);
            }
        }
    }
}
