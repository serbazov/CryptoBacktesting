class Backtrading:

    def __init__(self, cash, commission, equity=0):
        self.cash = cash  # баланс в деньгах
        self.equity = equity  # кол-во манеток в портфеле
        self.commission = commission  # комиссия
        # self.buy_trades_x = []
        # self.buy_trades_y = []
        # self.sell_trades_x = []
        # self.sell_trades_y = []

    def buy(self, quantity, price):
        self.cash = self.cash - price * quantity - price * quantity * self.commission
        self.equity = self.equity + quantity

    def sell(self, quantity, price):
        self.cash = self.cash + price * quantity - price * quantity * self.commission
        self.equity = self.equity - quantity

    #def imitation_step(self,step):