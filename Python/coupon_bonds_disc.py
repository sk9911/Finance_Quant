class CouponBonds:
    def __init__(self, principal, coupon_rate, interest_rate, maturity):
        self.principal = principal
        self.coupon = (coupon_rate/100)*principal
        self.interest_rate = interest_rate/100
        self.maturity = maturity

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

        price = self.present_value(self.principal, self.maturity)

        for t in range(1,self.maturity+1):
            price += self.present_value(self.coupon, t)

        return price


if __name__ == '__main__':
    bond = CouponBonds(1000, 10, 4, 3)
    print(f"Price of the bond = {bond.calculate_price():.2f}")