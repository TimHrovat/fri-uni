{ }
i = 0 ;
{ i = 0 }
m = 1 ;
{ i = 0 , m = 0 }
{ m = 2^i, i <= 32 }
while i < 32 do
    { m = 2^i, i < 32 }
    i := i + 1 ;
    { m = 2^(i-1), i - 1 < 32 }
    { m * 2 = 2^i, i - 1 < 32 }
    m := m * 2
    { m = 2^i, i <= 32 }
done
{ m = 2^i, i <= 32, i >= 32 }
{ m = 2^i, i = 32 }
{ m = 2^32 }
while i != 100 do
    { m = 2^32, i != 100 }
    i := i + 3 ;
    { m = 2^32 }
    if i >= m then
        { m = 2^32, i >= m }
        i := i - m
        { m = 2^32, i >= 0 }
    else
        { m = 2^32 }
        pass
        { m = 2^32 }
    end
    { m = 2^32 }
done
{ m = 2^32, i = 100 }
{ i = 100, m = 2^32 }
