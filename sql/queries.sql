-- Part A queries (SQLite 3.25+). Each block starts with "-- name: <id>" and is run by copilot/part_a.py.
-- Only completed orders count as revenue / spend. All amounts are INR.

-- name: q1_big_orders_having
SELECT customer_id, COUNT(*) AS big_orders, ROUND(SUM(order_amount), 2) AS total_inr
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_inr DESC;

-- name: q2_revenue_city_category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC;

-- name: q3_products_above_average_price
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: q4_customers_without_completed_order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: q5_above_average_spenders
WITH spend AS (
  SELECT customer_id, SUM(order_amount) AS total_spend
  FROM orders WHERE status = 'completed' GROUP BY customer_id),
avg_spend AS (SELECT AVG(total_spend) AS avg_total FROM spend)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend_inr, ROUND(a.avg_total, 2) AS average_spend_inr
FROM spend s CROSS JOIN avg_spend a
WHERE s.total_spend > a.avg_total
ORDER BY s.total_spend DESC;

-- name: q6_rank_within_city
WITH spend AS (
  SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
  FROM orders o JOIN customers c ON c.customer_id = o.customer_id
  WHERE o.status = 'completed' GROUP BY c.customer_id, c.city)
SELECT city, customer_id, ROUND(total_spend, 2) AS total_spend_inr,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM spend
ORDER BY city, spend_row_number;

-- name: q7_daily_revenue_running_total
WITH daily AS (
  SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
  FROM orders WHERE status = 'completed' GROUP BY day)
SELECT day, ROUND(revenue, 2) AS revenue_inr,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue_inr
FROM daily ORDER BY day;
