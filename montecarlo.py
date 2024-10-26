def get_data(stocks, start, end):
  stockData = pdr.get_data_yahoo(stocks, start, end)
  stockData = stockData['Close']
  returns = stockData.pct_change()
  meanReturns = returns.mean()
  covMatrix = returns.cov()

  return meanReturns, covMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Begin Monte Carlo method
mc_sims = 400 # num of simulations
T = 100 #timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
  Z = np.random.normal(size=(T, len(weights))) #uncorrelated RV's
  L = np.linalg.cholesky(covMatrix) #Cholesky decomposition to Lower Triangular
  dailyReturns = meanM = np.inner(L, Z) #Correlated daily returns for individual stocks
  portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1) * initialPortfolio

# Prepare monte carlo graph
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

# Outputs percentile on return distribution to a given confidence
# interal alpha
def mcVaR(returns, alpha=5):
  if isinstance(returns, pd.Series):
    return np.percentile(returns, alpha)
  else:
    raise TypeError("Expected a pandas data series")

#Outputs CVaR or Expected Shortfall to given confidence level alpha
def mcCVaR(returns, alpha=5):
  if isinstance(returns, pd.Series):
    belowVaR = returns <= mcVaR(returns, alpha=alpha)
    return returns[belowVaR].mean()
  else:
    raise TypeError("Expected a pandas data series")


portResults = pd.Series(portfolio_sims[-1,:])
VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

print('VaR_5 ${}'.format(round(VaR,2)))
print('CVaR_5 ${}'.format(round(CVaR,2)))

