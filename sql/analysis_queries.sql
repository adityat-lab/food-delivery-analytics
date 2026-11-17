-- Q1: Average delivery time by traffic density
SELECT traffic, COUNT(*) AS orders, ROUND(AVG(delivery_time_min), 1) AS avg_time
FROM orders
GROUP BY traffic
ORDER BY avg_time DESC;

-- Q2: Peak ordering hours
SELECT order_hour, COUNT(*) AS orders
FROM orders
GROUP BY order_hour
ORDER BY orders DESC
LIMIT 5;

-- Q3: Weather x traffic: average delivery time
SELECT weather, traffic, ROUND(AVG(delivery_time_min), 1) AS avg_time, COUNT(*) AS orders
FROM orders
WHERE weather != 'Unknown' AND traffic != 'Unknown'
GROUP BY weather, traffic
HAVING COUNT(*) >= 20
ORDER BY avg_time DESC
LIMIT 10;

-- Q4 (CTE): share of slow deliveries (> 40 min) by city type
WITH flagged AS (
    SELECT city, CASE WHEN delivery_time_min > 40 THEN 1 ELSE 0 END AS is_slow
    FROM orders
)
SELECT city, COUNT(*) AS orders, ROUND(100.0 * SUM(is_slow) / COUNT(*), 1) AS pct_slow
FROM flagged
GROUP BY city
ORDER BY pct_slow DESC;

-- Q5 (Window): rank vehicle types by speed inside each city type
SELECT city, vehicle_type, ROUND(AVG(delivery_time_min), 1) AS avg_time,
       RANK() OVER (PARTITION BY city ORDER BY AVG(delivery_time_min)) AS speed_rank
FROM orders
GROUP BY city, vehicle_type;

-- Q6 (Window): top 3 busiest riders per city type, with rating
WITH rider_stats AS (
    SELECT city, rider_id, COUNT(*) AS deliveries,
           ROUND(AVG(rider_rating), 2) AS rating,
           ROUND(AVG(delivery_time_min), 1) AS avg_time
    FROM orders
    GROUP BY city, rider_id
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY city ORDER BY deliveries DESC) AS rn
    FROM rider_stats
)
SELECT city, rider_id, deliveries, rating, avg_time
FROM ranked
WHERE rn <= 3;

-- Q7: festival vs non-festival performance
SELECT festival, COUNT(*) AS orders, ROUND(AVG(delivery_time_min), 1) AS avg_time,
       ROUND(AVG(distance_km), 2) AS avg_distance
FROM orders
GROUP BY festival;

-- Q8 (Window): daily orders with 7-day moving average
WITH daily AS (
    SELECT order_date, COUNT(*) AS orders
    FROM orders
    GROUP BY order_date
)
SELECT order_date, orders,
       ROUND(AVG(orders) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS moving_avg_7d
FROM daily
ORDER BY order_date
LIMIT 15;