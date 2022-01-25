class ZeroCouponBonds:
    def __init__(self, principal, maturity, interest_rate):
        self.principal = principal
        self.maturity = maturity
        self.interest_rate = interest_rate/100

    def present_value(self, x, n):
        """This function calculates the present value of a future payment

        Args:
            x (integer): Amount to be discounted
            n (integer): Time in years

        Returns:
            Integer: Returns the discounted value using x/(1+interest_rate)^n
        """
        return x/((1+self.interest_rate)**n)

    def calculate_price(self):
        """Calculates the price of a bond using current market interest rate

        Returns:
            Float: Calculates price of bond using Principal/(1+Rate)^T
        """
        return self.present_value(self.principal, self.maturity)

if __name__ == '__main__':
    bond = ZeroCouponBonds(1000, 2, 4)
    print(f"Price of the bond = {bond.calculate_price():.2f}")
